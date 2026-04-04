import matplotlib.pyplot as plt
import sqlite3 as sql
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy import special
from math import sqrt
import numpy as np
import utils
from scipy import ndimage
import pickle
from configuration import Config
from configuration import update_class_from_dict
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from microstructure import voronoi

# Display constants (avoid magic numbers)
DPI_FACTOR = 72.0
CM_PER_INCH = 1 / 2.54
FONT_NAME = "Times New Roman"
# Camera presets: (azim, elev, dist)
CAM_ANIM_SEPARATE = (-70, 30, 8)
CAM_ANIM_COMBINED = (-45, 22, 7.5)
CAM_PLOT3D_SEPARATE = (-92, 0, 8)
CAM_PLOT3D_COMBINED = (-131, 17, 2)
SIZE_UM_LABEL = "[µm]"


class Visualisation:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sql.connect(self.db_name)
        self.c = self.conn.cursor()
        self.Config = None
        self.microstructure = None
        self.axlim = None
        self.shape = None
        self.last_i = None
        self.oxid_numb = None
        self.utils = utils.Utils()
        self._table_prefix_aliases = {}
        self._available_iter_prefixes = set()
        self.generate_param_from_db()
        self.cell_size_full = 40
        self.cell_size = 40
        self.linewidth_f = 0.1
        self.linewidth = 0.2
        self.alpha = 1
        self.cm = {1: np.array([255, 200, 200]) / 255.0,
                   2: np.array([255, 75, 75]) / 255.0,
                   3: np.array([220, 0, 0]) / 255.0,
                   4: np.array([120, 0, 0]) / 255.0}

    # -------------------------------------------------------------------------
    # Helpers (reduce repetition and hardcoding)
    # -------------------------------------------------------------------------
    def _fetch_iter_table(self, iteration, table_prefix):
        """Load one iteration table as numpy array. Returns (N,3) or empty array on error."""
        try:
            table_prefix = self._resolve_table_prefix(table_prefix)
            self.c.execute('SELECT * from "{}_iter_{}"'.format(table_prefix, iteration))
            out = np.array(self.c.fetchall())
            return out if out.size else np.zeros((0, 3), dtype=np.int64)
        except (sql.OperationalError, TypeError):
            return np.zeros((0, 3), dtype=np.int64)

    def _build_table_prefix_aliases(self):
        self.c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_iter_%'")
        self._available_iter_prefixes = set()
        for (name,) in self.c.fetchall():
            if "_iter_" in name:
                self._available_iter_prefixes.add(name.rsplit("_iter_", 1)[0])

        aliases = {
            "primary_oxidant": str(getattr(self.Config.OXIDANTS.PRIMARY, "ELEMENT", "primary_oxidant")),
            "secondary_oxidant": str(getattr(self.Config.OXIDANTS.SECONDARY, "ELEMENT", "secondary_oxidant")),
            "primary_active": str(getattr(self.Config.ACTIVES.PRIMARY, "ELEMENT", "primary_active")),
            "secondary_active": str(getattr(self.Config.ACTIVES.SECONDARY, "ELEMENT", "secondary_active")),
            "primary_product": str(getattr(self.Config.PRODUCTS.PRIMARY, "ELEMENT", "primary_product")),
            "secondary_product": str(getattr(self.Config.PRODUCTS.SECONDARY, "ELEMENT", "secondary_product")),
            "ternary_product": str(getattr(self.Config.PRODUCTS.TERNARY, "ELEMENT", "ternary_product")),
            "quaternary_product": str(getattr(self.Config.PRODUCTS.QUATERNARY, "ELEMENT", "quaternary_product")),
            "quint_product": str(getattr(self.Config.PRODUCTS.QUINT, "ELEMENT", "quint_product")),
        }
        self._table_prefix_aliases = aliases

    def _resolve_table_prefix(self, table_prefix):
        if table_prefix in self._available_iter_prefixes:
            return table_prefix
        mapped = self._table_prefix_aliases.get(table_prefix, table_prefix)
        if mapped in self._available_iter_prefixes:
            return mapped
        return table_prefix

    def _scatter_size(self, fig, cell_size=40):
        """Marker size for 3D scatter from fig.dpi and cell_size."""
        return cell_size * (DPI_FACTOR / fig.dpi) ** 2

    def _rescale_factor(self):
        """Physical size (µm) / grid cells."""
        return (self.Config.SIZE * 1e6) / self.axlim

    def _primary_product_full_notfull(self, items):
        """From raw product rows, return (fulls, not_fulls) by oxidation number."""
        if items is None or not np.any(items):
            return np.zeros((0, 3)), np.zeros((0, 3))
        counts = np.unique(np.ravel_multi_index(items.transpose(), self.shape), return_counts=True)
        dec = np.array(np.unravel_index(counts[0], self.shape), dtype=np.short).transpose()
        cnt = np.array(counts[1], dtype=np.ubyte)
        full_ind = np.where(cnt == self.oxid_numb)[0]
        fulls = dec[full_ind]
        not_fulls = np.delete(dec, full_ind, axis=0)
        return fulls, not_fulls

    def _set_axes_lim_3d(self, axes, lim):
        """Set x/y/z lim for 3D axes (single or list)."""
        for ax in (axes if isinstance(axes, (list, tuple)) else [axes]):
            ax.set_xlim3d(0, lim)
            ax.set_ylim3d(0, lim)
            ax.set_zlim3d(0, lim)

    def _set_camera_3d(self, axes, azim, elev, dist):
        """Set view for 3D axes."""
        for ax in (axes if isinstance(axes, (list, tuple)) else [axes]):
            ax.azim = azim
            ax.elev = elev
            ax.dist = dist

    def _style_axis_times(self, ax, font_size_cm=60, labelpad=20):
        """Apply Times New Roman and tick/label styling to axis."""
        csfont = {"fontname": FONT_NAME}
        ax.tick_params(axis="x", labelsize=font_size_cm * CM_PER_INCH, labelcolor="black", pad=labelpad)
        ax.tick_params(axis="y", labelsize=font_size_cm * CM_PER_INCH, labelcolor="black", pad=labelpad)
        ax.tick_params(axis="z", labelsize=font_size_cm * CM_PER_INCH, labelcolor="black", pad=labelpad)
        for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            tick.set_fontname(FONT_NAME)
        ax.set_xlabel("X " + SIZE_UM_LABEL, **csfont, fontsize=font_size_cm * CM_PER_INCH, labelpad=labelpad)
        ax.set_ylabel("Y " + SIZE_UM_LABEL, **csfont, fontsize=font_size_cm * CM_PER_INCH, labelpad=labelpad)
        ax.set_zlabel("Z " + SIZE_UM_LABEL, **csfont, fontsize=font_size_cm * CM_PER_INCH, labelpad=labelpad)

    def _get_3d_panels(self):
        """Return list of (title, table, color, active) for separate 3D/2D panels."""
        cfg = self.Config
        pox = self._resolve_table_prefix("primary_oxidant")
        sox = self._resolve_table_prefix("secondary_oxidant")
        pact = self._resolve_table_prefix("primary_active")
        sact = self._resolve_table_prefix("secondary_active")
        pprod = self._resolve_table_prefix("primary_product")
        sprod = self._resolve_table_prefix("secondary_product")
        tprod = self._resolve_table_prefix("ternary_product")
        qprod = self._resolve_table_prefix("quaternary_product")
        return [
            (f"{pox} (inward diffusion)", pox, "b", bool(cfg.INWARD_DIFFUSION and pox in self._available_iter_prefixes)),
            (f"{sox} (inward diffusion)", sox, "deeppink", bool(cfg.INWARD_DIFFUSION and sox in self._available_iter_prefixes)),
            (f"{pact} (outward diffusion)", pact, "g", bool(cfg.OUTWARD_DIFFUSION and pact in self._available_iter_prefixes)),
            (f"{sact} (outward diffusion)", sact, "darkorange", bool(cfg.OUTWARD_DIFFUSION and sact in self._available_iter_prefixes)),
            (f"{pprod} (precipitation)", pprod, "r", bool(cfg.COMPUTE_PRECIPITATION and pprod in self._available_iter_prefixes)),
            (f"{sprod} (precipitation)", sprod, "cyan", bool(cfg.COMPUTE_PRECIPITATION and sprod in self._available_iter_prefixes)),
            (f"{tprod} (precipitation)", tprod, "darkgreen", bool(cfg.COMPUTE_PRECIPITATION and tprod in self._available_iter_prefixes)),
            (f"{qprod} (precipitation)", qprod, "steelblue", bool(cfg.COMPUTE_PRECIPITATION and qprod in self._available_iter_prefixes)),
        ]

    def generate_param_from_db(self):
        # Check if the db has an old layout
        table_name = 'PickledConfig'
        self.c.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        result = self.c.fetchone()

        if result is None:
            self.generate_config_from_old_db()
        else:
            self.c.execute("SELECT pickled_data FROM PickledConfig")
            result = self.c.fetchone()
            pickled_instance = result[0]
            unpickled_dict = pickle.loads(pickled_instance)
            update_class_from_dict(Config, unpickled_dict)
            self.Config = Config()

        table_name = 'PickledMicrostructure'
        self.c.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        result = self.c.fetchone()
        if result is not None:
            self.c.execute("SELECT pickled_data FROM PickledMicrostructure")
            result = self.c.fetchone()
            pickled_instance = result[0]
            self.microstructure = pickle.loads(pickled_instance)
            self.microstructure.show_microstructure()

        self.utils.print_static_params(Config)
        self.c.execute("SELECT last_i from time_parameters")
        self.last_i = self.c.fetchone()[0]
        self.compute_elapsed_time()

        self.axlim = self.Config.N_CELLS_PER_AXIS
        self.shape = (self.axlim, self.axlim, self.axlim)
        self.oxid_numb = self.Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER
        self._build_table_prefix_aliases()

        if not self.Config.INWARD_DIFFUSION:
            print("No INWARD data!")
        if not self.Config.COMPUTE_PRECIPITATION:
            print("No PRECIPITATION data!")
        if not self.Config.OUTWARD_DIFFUSION:
            print("No OUTWARD data!")

    def generate_config_from_old_db(self):
        user_input = utils.DEFAULT_PARAM
        self.c.execute("SELECT * from user_input")
        user_input_from_db = self.c.fetchall()[0]
        for position, key in enumerate(user_input):
            if 2 < position < len(user_input_from_db) + 3:
                user_input[key] = user_input_from_db[position - 3]

        self.c.execute("SELECT * from element_0")
        elem_data_from_db = self.c.fetchall()[0]
        user_input["active_element"]["primary"]["elem"] = elem_data_from_db[0]
        user_input["active_element"]["primary"]["diffusion_condition"] = elem_data_from_db[1]
        user_input["active_element"]["primary"]["mass_concentration"] = elem_data_from_db[2]
        user_input["active_element"]["primary"]["cells_concentration"] = elem_data_from_db[3]
        self.c.execute("SELECT * from element_1")
        elem_data_from_db = self.c.fetchall()[0]
        user_input["active_element"]["secondary"]["elem"] = elem_data_from_db[0]
        user_input["active_element"]["secondary"]["diffusion_condition"] = elem_data_from_db[1]
        user_input["active_element"]["secondary"]["mass_concentration"] = elem_data_from_db[2]
        user_input["active_element"]["secondary"]["cells_concentration"] = elem_data_from_db[3]
        self.c.execute("SELECT * from element_2")
        elem_data_from_db = self.c.fetchall()[0]
        user_input["oxidant"]["primary"]["elem"] = elem_data_from_db[0]
        user_input["oxidant"]["primary"]["diffusion_condition"] = elem_data_from_db[1]
        user_input["oxidant"]["primary"]["cells_concentration"] = elem_data_from_db[2]
        self.c.execute("SELECT * from element_3")
        elem_data_from_db = self.c.fetchall()[0]
        user_input["oxidant"]["secondary"]["elem"] = elem_data_from_db[0]
        user_input["oxidant"]["secondary"]["diffusion_condition"] = elem_data_from_db[1]
        user_input["oxidant"]["secondary"]["cells_concentration"] = elem_data_from_db[2]
        self.c.execute("SELECT * from element_4")
        elem_data_from_db = self.c.fetchall()[0]
        user_input["matrix_elem"]["elem"] = elem_data_from_db[0]
        user_input["matrix_elem"]["diffusion_condition"] = elem_data_from_db[1]
        user_input["matrix_elem"]["concentration"] = elem_data_from_db[2]

        # oxidants
        # _______________________________________________________________________________
        # primary
        Config.OXIDANTS.PRIMARY.ELEMENT = user_input["oxidant"]["primary"]["elem"]
        Config.OXIDANTS.PRIMARY.DIFFUSION_CONDITION = user_input["oxidant"]["primary"]["diffusion_condition"]
        Config.OXIDANTS.PRIMARY.CELLS_CONCENTRATION = user_input["oxidant"]["primary"]["cells_concentration"]
        # secondary
        Config.OXIDANTS.SECONDARY.ELEMENT = user_input["oxidant"]["secondary"]["elem"]
        Config.OXIDANTS.SECONDARY.DIFFUSION_CONDITION = user_input["oxidant"]["secondary"]["diffusion_condition"]
        Config.OXIDANTS.SECONDARY.CELLS_CONCENTRATION = user_input["oxidant"]["secondary"]["cells_concentration"]
        # _______________________________________________________________________________

        # actives
        # _______________________________________________________________________________
        # primary
        Config.ACTIVES.PRIMARY.ELEMENT = user_input["active_element"]["primary"]["elem"]
        Config.ACTIVES.PRIMARY.DIFFUSION_CONDITION = user_input["active_element"]["primary"]["diffusion_condition"]
        Config.ACTIVES.PRIMARY.MASS_CONCENTRATION = user_input["active_element"]["primary"]["mass_concentration"]
        Config.ACTIVES.PRIMARY.CELLS_CONCENTRATION = user_input["active_element"]["primary"]["cells_concentration"]
        # secondary
        Config.ACTIVES.SECONDARY.ELEMENT = user_input["active_element"]["secondary"]["elem"]
        Config.ACTIVES.SECONDARY.DIFFUSION_CONDITION = user_input["active_element"]["secondary"]["diffusion_condition"]
        Config.ACTIVES.SECONDARY.MASS_CONCENTRATION = user_input["active_element"]["secondary"]["mass_concentration"]
        Config.ACTIVES.SECONDARY.CELLS_CONCENTRATION = user_input["active_element"]["secondary"]["cells_concentration"]
        # _______________________________________________________________________________

        # matrix
        # _______________________________________________________________________________
        Config.MATRIX.ELEMENT = user_input["matrix_elem"]["elem"]
        # _______________________________________________________________________________

        Config.TEMPERATURE = user_input["temperature"]
        Config.N_CELLS_PER_AXIS = user_input["n_cells_per_axis"]
        Config.N_ITERATIONS = user_input["n_iterations"]
        Config.STRIDE = user_input["stride"]
        Config.STRIDE_MULTIPLIER = "WAS NOT IMPLEMENTED AT THAT TIME"
        Config.PRECIP_TRANSFORM_DEPTH = "WAS NOT IMPLEMENTED AT THAT TIME"
        Config.SIM_TIME = user_input["sim_time"]
        Config.SIZE = user_input["size"]
        Config.SOL_PROD = user_input["sol_prod"]
        Config.PHASE_FRACTION_LIMIT = user_input["phase_fraction_lim"]

        Config.THRESHOLD_INWARD = user_input["threshold_inward"]
        Config.THRESHOLD_OUTWARD = user_input["threshold_outward"]
        Config.NEIGH_RANGE = user_input["neigh_range"]

        Config.ROD_INCR_CONST = 0
        Config.ZETTA_ZERO = 0
        Config.ZETTA_FINAL = 0

        Config.INWARD_DIFFUSION = user_input["inward_diffusion"]
        Config.OUTWARD_DIFFUSION = user_input["outward_diffusion"]
        Config.COMPUTE_PRECIPITATION = user_input["compute_precipitations"]
        Config.SAVE_WHOLE = user_input["save_whole"]
        Config.DECOMPOSE_PRECIPITATIONS = user_input["decompose_precip"]
        Config.FULL_CELLS = user_input["full_cells"]
        Config.SAVE_PATH = user_input["save_path"]
        Config.SAVE_POST_PROCESSED_INPUT = False

        # PROBABILITIES
        # _______________________________________________________________________________
        # primary
        # nucleation
        # _________________________
        Config.PROBABILITIES.PRIMARY.p0 = user_input["nucleation_probability"]
        Config.PROBABILITIES.PRIMARY.p0_f = user_input["final_nucl_prob"]
        Config.PROBABILITIES.PRIMARY.p0_A_const = "LOST"
        Config.PROBABILITIES.PRIMARY.p0_B_const = user_input["b_const_P0_nucl"]

        Config.PROBABILITIES.PRIMARY.p1 = user_input["init_P1"]
        Config.PROBABILITIES.PRIMARY.p1_f = user_input["final_P1"]
        Config.PROBABILITIES.PRIMARY.p1_A_const = "LOST"
        Config.PROBABILITIES.PRIMARY.p1_B_const = user_input["b_const_P1"]

        Config.PROBABILITIES.PRIMARY.global_A = "LOST"
        Config.PROBABILITIES.PRIMARY.global_B = user_input["bend_b_init"]
        Config.PROBABILITIES.PRIMARY.global_B_f = user_input["bend_b_final"]

        Config.PROBABILITIES.PRIMARY.max_neigh_numb = user_input["max_neigh_numb"]
        Config.PROBABILITIES.PRIMARY.nucl_adapt_function = user_input["nucl_adapt_function"]
        # _________________________
        # dissolution
        # _________________________
        Config.PROBABILITIES.PRIMARY.p0_d = user_input["dissolution_p"]
        Config.PROBABILITIES.PRIMARY.p0_d_f = user_input["final_dissol_prob"]
        Config.PROBABILITIES.PRIMARY.p0_d_A_const = "LOST"
        Config.PROBABILITIES.PRIMARY.p0_d_B_const = "LOST"

        Config.PROBABILITIES.PRIMARY.p1_d = user_input["init_P1_diss"]
        Config.PROBABILITIES.PRIMARY.p1_d_f = user_input["final_P1_diss"]
        Config.PROBABILITIES.PRIMARY.p1_d_A_const = "LOST"
        Config.PROBABILITIES.PRIMARY.p1_d_B_const = user_input["b_const_P1_diss"]

        Config.PROBABILITIES.PRIMARY.p6_d = user_input["min_dissol_prob"]
        Config.PROBABILITIES.PRIMARY.p6_d_f = user_input["final_min_dissol_prob"]
        Config.PROBABILITIES.PRIMARY.p6_d_A_const = "LOST"
        Config.PROBABILITIES.PRIMARY.p6_d_B_const = "LOST"

        Config.PROBABILITIES.PRIMARY.global_d_A = "LOST"
        Config.PROBABILITIES.PRIMARY.global_d_B = "LOST"
        Config.PROBABILITIES.PRIMARY.global_d_B_f = "LOST"

        Config.PROBABILITIES.PRIMARY.bsf = "LOST"
        Config.PROBABILITIES.PRIMARY.dissol_adapt_function = "LOST"
        # ________________________
        Config.COMMENT = """NO COMMENTS"""
        Config.INITIAL_SCRIPT = "LOST"

        self.utils.generate_param()
        self.Config = Config()

    def compute_elapsed_time(self):
        self.c.execute("SELECT elapsed_time from time_parameters")
        elapsed_time_sek = np.array(self.c.fetchall()[0])
        if elapsed_time_sek != 0:
            h = elapsed_time_sek // 3600
            m = (elapsed_time_sek - h * 3600) // 60
            s = elapsed_time_sek - h * 3600 - m * 60
            message = f'{int(h)}h:{int(m)}m:{int(s)}s'
        else:
            message = f'Simulation was interrupted at iteration = {self.last_i}'

        print(f"""
TIME:------------------------------------------------------------
ELAPSED TIME: {message}
-----------------------------------------------------------------""")

    def animate_3d(self, animate_separate=False, const_cam_pos=False):
        if not self.Config.SAVE_WHOLE:
            return print("No Data To Animate!")

        def animate(iteration):
            ax_all.cla()
            ax_all.dist = 4
            s = self._scatter_size(fig, self.cell_size)
            if self.Config.INWARD_DIFFUSION:
                items = self._fetch_iter_table(iteration, "primary_oxidant")
                if np.any(items):
                    ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='b', s=s)
                if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    items = self._fetch_iter_table(iteration, "secondary_oxidant")
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='deeppink', s=s)
            if self.Config.OUTWARD_DIFFUSION:
                items = self._fetch_iter_table(iteration, "primary_active")
                if np.any(items):
                    ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='g', s=s)
                if self.Config.ACTIVES.SECONDARY_EXISTENCE:
                    items = self._fetch_iter_table(iteration, "secondary_active")
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkorange', s=s)
            if self.Config.COMPUTE_PRECIPITATION:
                items = self._fetch_iter_table(iteration, "primary_product")
                if np.any(items):
                    fulls, not_fulls = self._primary_product_full_notfull(items)
                    ax_all.scatter(fulls[:, 2], fulls[:, 1], fulls[:, 0], marker=',', color="darkred", s=s,
                                   edgecolors='black', linewidth=self.linewidth, alpha=self.alpha)
                    ax_all.scatter(not_fulls[:, 2], not_fulls[:, 1], not_fulls[:, 0], marker=',', color='darkred', s=s,
                                   edgecolors='black', linewidth=self.linewidth, alpha=self.alpha)
                if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    for table, color in [("secondary_product", "cyan"), ("ternary_product", "darkorange"),
                                         ("quaternary_product", "steelblue"), ("quint_product", "darkviolet")]:
                        items = self._fetch_iter_table(iteration, table)
                        if np.any(items):
                            ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color=color, s=s,
                                           edgecolors='black', linewidth=self.linewidth)
                elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    items = self._fetch_iter_table(iteration, "secondary_product")
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan', s=s)

            self._set_axes_lim_3d(ax_all, self.axlim)
            if const_cam_pos:
                azim, elev, dist = CAM_ANIM_COMBINED
                self._set_camera_3d(ax_all, azim, elev, dist)

        if animate_separate:
            azim, elev, dist = CAM_ANIM_SEPARATE
            panels = []
            for title, table, color, active in self._get_3d_panels():
                if not active:
                    continue
                fig_i = plt.figure()
                fig_i.canvas.manager.set_window_title(title)
                ax = fig_i.add_subplot(111, projection='3d')
                panels.append((fig_i, ax, table, color))

            def make_updater(ax, table, color):
                def upd(iteration):
                    ax.cla()
                    s = self._scatter_size(ax.figure, self.cell_size)
                    items = self._fetch_iter_table(iteration, table)
                    if table == "primary_product" and np.any(items):
                        fulls, not_fulls = self._primary_product_full_notfull(items)
                        ax.scatter(fulls[:, 2], fulls[:, 1], fulls[:, 0], marker=',', color='darkred', s=s, edgecolors='black', linewidth=self.linewidth, alpha=self.alpha)
                        ax.scatter(not_fulls[:, 2], not_fulls[:, 1], not_fulls[:, 0], marker=',', color='darkred', s=s, edgecolors='black', linewidth=self.linewidth, alpha=self.alpha)
                    elif np.any(items):
                        ax.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color=color, s=min(s, 3) if table == "primary_active" else s)
                    self._set_axes_lim_3d(ax, self.axlim)
                    if const_cam_pos:
                        self._set_camera_3d(ax, azim, elev, dist)
                return upd

            for fig_i, ax, table, color in panels:
                FuncAnimation(fig_i, make_updater(ax, table, color))
            plt.show()
            plt.close('all')
            return

        fig = plt.figure()
        ax_all = fig.add_subplot(111, projection='3d')
        FuncAnimation(fig, animate)
        plt.show()

    def plot_3d(self, plot_separate=False, iteration=None, const_cam_pos=False):
        if iteration is None:
            iteration = self.last_i
        rescale_factor = self._rescale_factor()
        new_axlim = self.axlim * rescale_factor
        if plot_separate:
            kw = dict(edgecolors='black', linewidth=self.linewidth, alpha=self.alpha)
            for title, table, color, active in self._get_3d_panels():
                if not active:
                    continue
                items = self._fetch_iter_table(iteration, table)
                if not np.any(items):
                    continue
                fig = plt.figure()
                fig.canvas.manager.set_window_title(title)
                ax = fig.add_subplot(111, projection='3d')
                s_cell = self._scatter_size(fig, self.cell_size)
                if table == "primary_product":
                    fulls, not_fulls = self._primary_product_full_notfull(items)
                    fulls = np.asarray(fulls, dtype=float) * rescale_factor
                    not_fulls = np.asarray(not_fulls, dtype=float) * rescale_factor
                    ax.scatter(fulls[:, 2], fulls[:, 1], fulls[:, 0], marker=',', color='darkred', s=self._scatter_size(fig, self.cell_size_full), **kw)
                    ax.scatter(not_fulls[:, 2], not_fulls[:, 1], not_fulls[:, 0], marker=',', color='darkred', s=s_cell, **kw)
                else:
                    items_phys = np.asarray(items, dtype=float) * rescale_factor
                    ax.scatter(items_phys[:, 2], items_phys[:, 1], items_phys[:, 0], marker=',', color=color, s=s_cell, **kw)
                ax.set_title(title, fontname=FONT_NAME)
                ax.set_xlim3d(0, new_axlim)
                ax.set_ylim3d(0, new_axlim)
                ax.set_zlim3d(0, new_axlim)
                if const_cam_pos:
                    azim, elev, dist = CAM_PLOT3D_SEPARATE
                    self._set_camera_3d(ax, azim, elev, dist)
                step = new_axlim / 5
                ticks = np.arange(0, new_axlim + rescale_factor, step)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_zticks(ticks)
                self._style_axis_times(ax, font_size_cm=60, labelpad=20)
            plt.show()
            plt.close('all')
            return
        fig = plt.figure()
        s_cell = self._scatter_size(fig, self.cell_size)
        s_cell_full = self._scatter_size(fig, self.cell_size_full)
        ax_all = fig.add_subplot(111, projection='3d')
        kw = dict(edgecolors='black', linewidth=self.linewidth, alpha=self.alpha)
        if self.Config.INWARD_DIFFUSION:
            items = self._fetch_iter_table(iteration, "primary_oxidant")
            if np.any(items):
                items = np.asarray(items, dtype=float) * rescale_factor
                ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='b', s=s_cell, **kw)
            if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                items = self._fetch_iter_table(iteration, "secondary_oxidant")
                if np.any(items):
                    items = np.asarray(items, dtype=float) * rescale_factor
                    ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='deeppink', s=s_cell, **kw)
        if self.Config.OUTWARD_DIFFUSION:
            items = self._fetch_iter_table(iteration, "primary_active")
            if np.any(items):
                items = np.asarray(items, dtype=float) * rescale_factor
                ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='g', s=s_cell, **kw)
            if self.Config.ACTIVES.SECONDARY_EXISTENCE:
                items = self._fetch_iter_table(iteration, "secondary_active")
                if np.any(items):
                    items = np.asarray(items, dtype=float) * rescale_factor
                    ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkorange', s=s_cell, **kw)
        if self.Config.COMPUTE_PRECIPITATION:
            items = self._fetch_iter_table(iteration, "primary_product")
            if np.any(items):
                fulls, not_fulls = self._primary_product_full_notfull(items)
                fulls = np.asarray(fulls, dtype=float) * rescale_factor
                not_fulls = np.asarray(not_fulls, dtype=float) * rescale_factor
                ax_all.scatter(fulls[:, 2], fulls[:, 1], fulls[:, 0], marker=',', color="darkred", s=s_cell_full,
                               edgecolors='black', linewidth=self.linewidth_f, alpha=self.alpha)
                ax_all.scatter(not_fulls[:, 2], not_fulls[:, 1], not_fulls[:, 0], marker=',', color='darkred', s=s_cell, **kw)
            if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                for table, color in [("secondary_product", "cyan"), ("ternary_product", "darkorange"),
                                     ("quaternary_product", "steelblue"), ("quint_product", "darkviolet")]:
                    items = self._fetch_iter_table(iteration, table)
                    if np.any(items):
                        items = np.asarray(items, dtype=float) * rescale_factor
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color=color, s=s_cell, **kw)
            elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                items = self._fetch_iter_table(iteration, "secondary_product")
                if np.any(items):
                    items = np.asarray(items, dtype=float) * rescale_factor
                    ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='tomato', s=s_cell, **kw)
        ax_all.set_xlim3d(0, new_axlim)
        ax_all.set_ylim3d(0, new_axlim)
        ax_all.set_zlim3d(0, new_axlim)
        if const_cam_pos:
            azim, elev, dist = CAM_PLOT3D_COMBINED
            self._set_camera_3d(ax_all, azim, elev, dist)
        step = new_axlim / 5
        ticks = np.arange(0, new_axlim + rescale_factor, step)
        ax_all.set_xticks(ticks)
        ax_all.set_yticks(ticks)
        ax_all.set_zticks(ticks)
        self._style_axis_times(ax_all, font_size_cm=60, labelpad=20)
        plt.show()
        plt.close()

    def plot_2d(self, plot_separate=False, iteration=None, slice_pos=None):
        if iteration is None:
            iteration = self.last_i
        if slice_pos is None:
            slice_pos = int(self.axlim / 2)
        rescale_factor = self._rescale_factor()
        new_axlim = self.axlim * rescale_factor
        fig = plt.figure()
        s_cell = self._scatter_size(fig, self.cell_size)
        s_cell_full = self._scatter_size(fig, self.cell_size_full)
        if plot_separate:
            for title, table, color, active in self._get_3d_panels():
                if not active:
                    continue
                items = self._fetch_iter_table(iteration, table)
                if not np.any(items):
                    continue
                ind = np.where(items[:, 0] == slice_pos)[0]
                if len(ind) == 0:
                    continue
                fig_i = plt.figure()
                fig_i.canvas.manager.set_window_title(title)
                ax = fig_i.add_subplot(111)
                pts = np.asarray(items[ind], dtype=float) * rescale_factor
                if table == "primary_product":
                    slice_items = items[ind]
                    counts = np.unique(np.ravel_multi_index(slice_items.T, self.shape), return_counts=True)
                    dec = np.array(np.unravel_index(counts[0], self.shape), dtype=float).T
                    cnt = np.array(counts[1], dtype=np.ubyte)
                    full_ind = np.where(cnt == self.oxid_numb)[0]
                    fulls = dec[full_ind] * rescale_factor
                    not_fulls = np.delete(dec, full_ind, axis=0) * rescale_factor
                    ax.scatter(fulls[:, 2], fulls[:, 1], marker=',', color='darkred', s=self._scatter_size(fig_i, self.cell_size_full), edgecolors='black', linewidth=self.linewidth)
                    ax.scatter(not_fulls[:, 2], not_fulls[:, 1], marker=',', color='darkred', s=self._scatter_size(fig_i, self.cell_size), edgecolors='black', linewidth=self.linewidth)
                else:
                    ax.scatter(pts[:, 2], pts[:, 1], marker=',', color=color, s=self._scatter_size(fig_i, self.cell_size), edgecolors='black', linewidth=self.linewidth)
                ax.set_title(title, fontname=FONT_NAME)
                ax.set_xlim(-rescale_factor, (self.axlim * rescale_factor) + rescale_factor)
                ax.set_ylim(-rescale_factor, (self.axlim * rescale_factor) + rescale_factor)
                step = new_axlim / 5
                ticks = np.arange(0, new_axlim + 1, step)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                csfont = {'fontname': FONT_NAME}
                f_size = 50
                ax.tick_params(axis='x', labelsize=f_size * CM_PER_INCH, labelcolor='black', pad=1)
                ax.tick_params(axis='y', labelsize=f_size * CM_PER_INCH, labelcolor='black', pad=1)
                for tick in ax.get_xticklabels() + ax.get_yticklabels():
                    tick.set_fontname(FONT_NAME)
                ax.set_xlabel("X " + SIZE_UM_LABEL, **csfont, fontsize=f_size * CM_PER_INCH, labelpad=1)
                ax.set_ylabel("Y " + SIZE_UM_LABEL, **csfont, fontsize=f_size * CM_PER_INCH, labelpad=1)
            plt.show()
            plt.close('all')
            return
        ax_all = fig.add_subplot(111)
        ax_all.set_facecolor('gainsboro')
        if self.Config.INWARD_DIFFUSION:
            items = self._fetch_iter_table(iteration, "primary_oxidant")
            if np.any(items):
                ind = np.where(items[:, 0] == slice_pos)[0]
                if len(ind):
                    pts = np.asarray(items[ind], dtype=float) * rescale_factor
                    ax_all.scatter(pts[:, 2], pts[:, 1], marker=',', color='b', s=s_cell_full, edgecolors='black', linewidth=self.linewidth)
            if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                items = self._fetch_iter_table(iteration, "secondary_oxidant")
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)[0]
                    if len(ind):
                        pts = np.asarray(items[ind], dtype=float) * rescale_factor
                        ax_all.scatter(pts[:, 2], pts[:, 1], marker=',', color='deeppink', s=s_cell_full)
        if self.Config.OUTWARD_DIFFUSION:
            items = self._fetch_iter_table(iteration, "primary_active")
            if np.any(items):
                ind = np.where(items[:, 0] == slice_pos)[0]
                if len(ind):
                    pts = np.asarray(items[ind], dtype=float) * rescale_factor
                    ax_all.scatter(pts[:, 2], pts[:, 1], marker=',', color='g', s=s_cell_full, edgecolors='black', linewidth=self.linewidth)
            if self.Config.ACTIVES.SECONDARY_EXISTENCE:
                items = self._fetch_iter_table(iteration, "secondary_active")
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)[0]
                    if len(ind):
                        pts = np.asarray(items[ind], dtype=float) * rescale_factor
                        ax_all.scatter(pts[:, 2], pts[:, 1], marker=',', color='navy', s=s_cell_full)

        if self.Config.COMPUTE_PRECIPITATION:
            items = self._fetch_iter_table(iteration, "primary_product")
            if np.any(items):
                ind = np.where(items[:, 0] == slice_pos)[0]
                if len(ind):
                    slice_items = items[ind]
                    counts = np.unique(np.ravel_multi_index(slice_items.T, self.shape), return_counts=True)
                    dec = np.array(np.unravel_index(counts[0], self.shape), dtype=float).T
                    cnt = np.array(counts[1], dtype=np.ubyte)
                    full_ind = np.where(cnt == self.oxid_numb)[0]
                    fulls = dec[full_ind] * rescale_factor
                    not_fulls = np.delete(dec, full_ind, axis=0) * rescale_factor
                    ax_all.scatter(fulls[:, 2], fulls[:, 1], marker=',', color='darkred', s=s_cell_full, edgecolors='black', linewidth=self.linewidth)
                    ax_all.scatter(not_fulls[:, 2], not_fulls[:, 1], marker=',', color='darkred', s=s_cell, edgecolors='black', linewidth=self.linewidth)
            if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                for table, color in [("secondary_product", "cyan"), ("ternary_product", "darkorange"),
                                     ("quaternary_product", "steelblue"), ("quint_product", "darkviolet")]:
                    items = self._fetch_iter_table(iteration, table)
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)[0]
                        if len(ind):
                            pts = np.asarray(items[ind], dtype=float) * rescale_factor
                            ax_all.scatter(pts[:, 2], pts[:, 1], marker=',', color=color, s=s_cell, edgecolors='black', linewidth=self.linewidth)
            elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                items = self._fetch_iter_table(iteration, "secondary_product")
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)[0]
                    if len(ind):
                        pts = np.asarray(items[ind], dtype=float) * rescale_factor
                        ax_all.scatter(pts[:, 2], pts[:, 1], marker=',', color='cyan', s=s_cell)

        fig.set_size_inches((20 * CM_PER_INCH, 20 * CM_PER_INCH))
        step = new_axlim / 5
        ticks = np.arange(0, new_axlim + 1, step)
        ax_all.set_xticks(ticks)
        ax_all.set_yticks(ticks)
        f_size = 50
        ax_all.tick_params(axis='x', labelsize=f_size * CM_PER_INCH, labelcolor='black', pad=1)
        ax_all.tick_params(axis='y', labelsize=f_size * CM_PER_INCH, labelcolor='black', pad=1)
        for tick in ax_all.get_xticklabels() + ax_all.get_yticklabels():
            tick.set_fontname(FONT_NAME)
        csfont = {'fontname': FONT_NAME}
        ax_all.set_xlabel("X " + SIZE_UM_LABEL, **csfont, fontsize=f_size * CM_PER_INCH, labelpad=1)
        ax_all.set_ylabel("Y " + SIZE_UM_LABEL, **csfont, fontsize=f_size * CM_PER_INCH, labelpad=1)
        ax_all.set_xlim(-rescale_factor, (self.axlim * rescale_factor)+rescale_factor)
        ax_all.set_ylim(-rescale_factor, (self.axlim * rescale_factor)+rescale_factor)
        self.conn.commit()
        # plt.savefig(f'W:/SIMCA/test_runs_data/{slice_pos}.jpeg')
        # plt.savefig(f"//juno/homes/user/aseregin/Desktop/Neuer Ordner/{slice_pos}.jpeg")
        # plt.savefig(f'C:/test_runs_data/{slice_pos}.jpeg')
        plt.show()

    def animate_2d(self, plot_separate=False, slice_pos=None):
        if not self.Config.SAVE_WHOLE:
            print("No Data To Animate!")
            return
        if slice_pos is None:
            slice_pos = int(self.axlim / 2)

        def _slice_scatter(ax, items, color, s):
            if not np.any(items):
                return
            ind = np.where(items[:, 0] == slice_pos)[0]
            if len(ind) == 0:
                return
            ax.scatter(items[ind, 2], items[ind, 1], marker=',', color=color, s=s)

        if plot_separate:
            panels = []
            for title, table, color, active in self._get_3d_panels():
                if not active:
                    continue
                fig_i = plt.figure()
                fig_i.canvas.manager.set_window_title(title)
                ax = fig_i.add_subplot(111)
                panels.append((fig_i, ax, table, color))

            def make_updater_2d(ax, table, color):
                def upd(iteration):
                    ax.cla()
                    s = self._scatter_size(ax.figure, self.cell_size)
                    items = self._fetch_iter_table(iteration, table)
                    if table == "primary_product" and np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)[0]
                        if len(ind):
                            slice_items = items[ind]
                            counts = np.unique(np.ravel_multi_index(slice_items.T, self.shape), return_counts=True)
                            dec = np.array(np.unravel_index(counts[0], self.shape), dtype=float).T
                            cnt = np.array(counts[1], dtype=np.ubyte)
                            full_ind = np.where(cnt == self.oxid_numb)[0]
                            fulls = dec[full_ind]
                            not_fulls = np.delete(dec, full_ind, axis=0)
                            ax.scatter(fulls[:, 2], fulls[:, 1], marker=',', color='darkred', s=s)
                            ax.scatter(not_fulls[:, 2], not_fulls[:, 1], marker=',', color='darkred', s=s)
                    else:
                        _slice_scatter(ax, items, color, s)
                    ax.set_xlim(0, self.axlim)
                    ax.set_ylim(0, self.axlim)
                return upd

            for fig_i, ax, table, color in panels:
                FuncAnimation(fig_i, make_updater_2d(ax, table, color))
            plt.show()
            plt.close('all')
            return

        def animate(iteration):
            ax_all.cla()
            s = self._scatter_size(fig, self.cell_size)
            if self.Config.INWARD_DIFFUSION:
                items = self._fetch_iter_table(iteration, "primary_oxidant")
                _slice_scatter(ax_all, items, 'b', s)
            if self.Config.OUTWARD_DIFFUSION:
                items = self._fetch_iter_table(iteration, "primary_active")
                _slice_scatter(ax_all, items, 'g', s)
            if self.Config.COMPUTE_PRECIPITATION:
                items = self._fetch_iter_table(iteration, "primary_product")
                _slice_scatter(ax_all, items, 'r', s)
            ax_all.set_xlim(0, self.axlim)
            ax_all.set_ylim(0, self.axlim)

        fig = plt.figure()
        ax_all = fig.add_subplot(111)
        FuncAnimation(fig, animate)
        plt.show()

    def animate_concentration(self, analytic_sol=False, conc_type="atomic"):
        def animate(iteration):
            inward = np.zeros(self.axlim, dtype=int)
            inward_moles = np.zeros(self.axlim, dtype=int)
            inward_mass = np.zeros(self.axlim, dtype=int)

            sinward = np.zeros(self.axlim, dtype=int)
            sinward_moles = np.zeros(self.axlim, dtype=int)
            sinward_mass = np.zeros(self.axlim, dtype=int)

            outward = np.zeros(self.axlim, dtype=int)
            outward_moles = np.zeros(self.axlim, dtype=int)
            outward_mass = np.zeros(self.axlim, dtype=int)
            outward_eq_mat_moles = np.zeros(self.axlim, dtype=int)

            soutward = np.zeros(self.axlim, dtype=int)
            soutward_moles = np.zeros(self.axlim, dtype=int)
            soutward_mass = np.zeros(self.axlim, dtype=int)
            soutward_eq_mat_moles = np.zeros(self.axlim, dtype=int)

            primary_product = np.zeros(self.axlim, dtype=int)
            primary_product_moles = np.zeros(self.axlim, dtype=int)
            primary_product_mass = np.zeros(self.axlim, dtype=int)
            primary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

            secondary_product = np.zeros(self.axlim, dtype=int)
            secondary_product_moles = np.zeros(self.axlim, dtype=int)
            secondary_product_mass = np.zeros(self.axlim, dtype=int)
            secondary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

            ternary_product = np.zeros(self.axlim, dtype=int)
            ternary_product_moles = np.zeros(self.axlim, dtype=int)
            ternary_product_mass = np.zeros(self.axlim, dtype=int)
            ternary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

            quaternary_product = np.zeros(self.axlim, dtype=int)
            quaternary_product_moles = np.zeros(self.axlim, dtype=int)
            quaternary_product_mass = np.zeros(self.axlim, dtype=int)
            quaternary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

            if self.Config.INWARD_DIFFUSION:
                items = self._fetch_iter_table(iteration, "primary_oxidant")
                inward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                inward_moles = inward * self.Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
                inward_mass = inward * self.Config.OXIDANTS.PRIMARY.MASS_PER_CELL

                if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    items = self._fetch_iter_table(iteration, "secondary_oxidant")
                    sinward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    sinward_moles = sinward * self.Config.OXIDANTS.SECONDARY.MOLES_PER_CELL
                    sinward_mass = sinward * self.Config.OXIDANTS.SECONDARY.MASS_PER_CELL

            if self.Config.OUTWARD_DIFFUSION:
                items = self._fetch_iter_table(iteration, "primary_active")
                outward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                outward_moles = outward * self.Config.ACTIVES.PRIMARY.MOLES_PER_CELL
                outward_mass = outward * self.Config.ACTIVES.PRIMARY.MASS_PER_CELL
                outward_eq_mat_moles = outward * self.Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

                if self.Config.ACTIVES.SECONDARY_EXISTENCE:
                    items = self._fetch_iter_table(iteration, "secondary_active")
                    soutward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    soutward_moles = soutward * self.Config.ACTIVES.SECONDARY.MOLES_PER_CELL
                    soutward_mass = soutward * self.Config.ACTIVES.SECONDARY.MASS_PER_CELL
                    soutward_eq_mat_moles = soutward * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

            if self.Config.COMPUTE_PRECIPITATION:
                items = self._fetch_iter_table(iteration, "primary_product")
                if np.any(items):
                    primary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    primary_product_moles = primary_product * self.Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
                    primary_product_mass = primary_product * self.Config.PRODUCTS.PRIMARY.MASS_PER_CELL
                    primary_product_eq_mat_moles = primary_product * self.Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL
            #
            #     if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
            #         self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
            #         items = np.array(self.c.fetchall())
            #         secondary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
            #         secondary_product_moles = secondary_product * self.Config.PRODUCTS.SECONDARY.MOLES_PER_CELL
            #         secondary_product_mass = secondary_product * self.Config.PRODUCTS.SECONDARY.MASS_PER_CELL
            #         secondary_product_eq_mat_moles = secondary_product * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL
            #
            #         self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
            #         items = np.array(self.c.fetchall())
            #         ternary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
            #         ternary_product_moles = ternary_product * self.Config.PRODUCTS.TERNARY.MOLES_PER_CELL
            #         ternary_product_mass = ternary_product * self.Config.PRODUCTS.TERNARY.MASS_PER_CELL
            #         ternary_product_eq_mat_moles = ternary_product * self.Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL
            #
            #         self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
            #         items = np.array(self.c.fetchall())
            #         quaternary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
            #         quaternary_product_moles = quaternary_product * self.Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL
            #         quaternary_product_mass = quaternary_product * self.Config.PRODUCTS.QUATERNARY.MASS_PER_CELL
            #         quaternary_product_eq_mat_moles = quaternary_product * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL
            #
            #     elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
            #         self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
            #         items = np.array(self.c.fetchall())
            #         if np.any(items):
            #             secondary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
            #             secondary_product_moles = secondary_product * self.Config.PRODUCTS.SECONDARY.MOLES_PER_CELL
            #             secondary_product_mass = secondary_product * self.Config.PRODUCTS.SECONDARY.MASS_PER_CELL
            #             secondary_product_eq_mat_moles = primary_product * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

            self.conn.commit()
            # primary_product_left = np.sum(primary_product[:44])
            # primary_product_right = np.sum(primary_product[44:])
            #
            # print("left: ", primary_product_left, " right: ", primary_product_right)

            # n_matrix_page = (self.axlim ** 2) * self.param["product"]["primary"]["oxidation_number"]
            n_matrix_page = (self.axlim ** 2)
            matrix = np.full(self.axlim, n_matrix_page)

            matrix_moles = matrix * self.Config.MATRIX.MOLES_PER_CELL - outward_eq_mat_moles \
                           - soutward_eq_mat_moles - primary_product_eq_mat_moles - secondary_product_eq_mat_moles \
                           - ternary_product_eq_mat_moles - quaternary_product_eq_mat_moles
            matrix_mass = matrix_moles * self.Config.MATRIX.MOLAR_MASS

            # matrix = (n_matrix_page - outward - soutward -
            #           primary_product - secondary_product - ternary_product - quaternary_product)
            # less_than_zero = np.where(matrix < 0)[0]
            # matrix[less_than_zero] = 0

            # matrix_moles = matrix * self.param["active_element"]["primary"]["eq_matrix_moles_per_cell"]
            # matrix_mass = matrix * self.param["active_element"]["primary"]["eq_matrix_mass_per_cell"]

            x = np.linspace(0, self.Config.SIZE, self.axlim)

            if conc_type.lower() == "atomic":
                whole_moles = matrix_moles + \
                              inward_moles + sinward_moles + \
                              outward_moles + soutward_moles + \
                              primary_product_moles + secondary_product_moles + \
                              ternary_product_moles + quaternary_product_moles
                whole_moles = np.maximum(whole_moles, 1e-100)

                inward = inward_moles * 100 / whole_moles
                sinward = sinward_moles * 100 / whole_moles
                outward = outward_moles * 100 / whole_moles
                soutward = soutward_moles * 100 / whole_moles

                primary_product = primary_product_moles * 100 / whole_moles
                secondary_product = secondary_product_moles * 100 / whole_moles
                ternary_product = ternary_product_moles * 100 / whole_moles
                quaternary_product = quaternary_product_moles * 100 / whole_moles

            elif conc_type.lower() == "cells":
                n_cells_page = max(self.axlim ** 2, 1)
                inward = inward * 100 / n_cells_page
                sinward = sinward * 100 / n_cells_page
                outward = outward * 100 / n_cells_page
                soutward = soutward * 100 / n_cells_page

                primary_product = primary_product * 100 / n_cells_page
                secondary_product = secondary_product * 100 / n_cells_page
                ternary_product = ternary_product * 100 / n_cells_page
                quaternary_product = quaternary_product * 100 / n_cells_page

            elif conc_type.lower() == "mass":
                whole_mass = matrix_mass + \
                             inward_mass + sinward_mass + \
                             outward_mass + soutward_mass + \
                             secondary_product_mass + primary_product_mass + \
                             ternary_product_mass + quaternary_product_mass
                whole_mass = np.maximum(whole_mass, 1e-100)

                inward = inward_mass * 100 / whole_mass
                sinward = sinward_mass * 100 / whole_mass
                outward = outward_mass * 100 / whole_mass
                soutward = soutward_mass * 100 / whole_mass

                primary_product = primary_product_mass * 100 / whole_mass
                secondary_product = secondary_product_mass * 100 / whole_mass
                ternary_product = ternary_product_mass * 100 / whole_mass
                quaternary_product = quaternary_product_mass * 100 / whole_mass

            else:
                print("WRONG CONCENTRATION TYPE!")

            ax1.cla()
            ax2.cla()
            ax1.plot(x, inward, color='b')
            ax1.plot(x, sinward, color='deeppink')

            ax2.plot(x, outward, color='g')
            ax2.plot(x, soutward, color='darkorange')

            ax2.plot(x, primary_product, color='r')
            ax2.plot(x, secondary_product, color='cyan')
            ax2.plot(x, ternary_product, color='darkgreen')
            ax2.plot(x, quaternary_product, color='steelblue')

            if analytic_sol:
                y_max = self.Config.OXIDANTS.PRIMARY.CELLS_CONCENTRATION * 100
                # y_max_out = self.param["active_elem_conc"] * 100

                diff_c = self.Config.OXIDANTS.PRIMARY.DIFFUSION_COEFFICIENT

                analytical_concentration_maxy =\
                    y_max * special.erfc(x / (2 * sqrt(diff_c * (iteration + 1) * self.Config.SIM_TIME / self.Config.N_ITERATIONS)))
                ax1.plot(x, analytical_concentration_maxy, color='r')

                # analytical_concentration_out = (y_max_out/2) * (1 - special.erf((- x) / (2 * sqrt(
                #     self.param["diff_coeff_out"] * (iteration + 1) * self.param["sim_time"] / self.param["n_iterations"]))))

                # proz = [sqrt((analytic - outw)**2) / analytic for analytic, outw in zip(analytical_concentration_out, outward)]
                # proz_mean = (np.sum(proz[0:10]) / 10) * 100
                # summa = analytical_concentration_out - outward
                # summa = np.sum(summa[0:10])
                # print(f"""{iteration} {proz_mean}""")

                # ax1.set_ylim(0, y_max_out + y_max_out * 0.2)
                # ax1.plot(x, analytical_concentration_out, color='r', linewidth=1.5)
            # if analytic_sol_sand:
            #     self.c.execute("SELECT y_max_sand from description")
            #     y_max_sand = self.c.fetchone()[0] / 2
            #     self.c.execute("SELECT half_thickness from description")
            #     half_thickness = self.c.fetchone()[0]
            #     # left = ((self.n_cells_per_axis / 2) - half_thickness) * self.lamda - self.lamda
            #     # right = ((self.n_cells_per_axis / 2) + half_thickness) * self.lamda + self.lamda
            #
            #     #  for point!
            #     # left = int(self.n_cells_per_axis / 2) * self.lamda
            #     # right = (int(self.n_cells_per_axis / 2) + half_thickness) * self.lamda
            #
            #     left = (int(self.param["n_cells_per_axis"]n_cells_per_axis / 2) - half_thickness) * self.param["l_ambda"]
            #     right = (int(self.param["n_cells_per_axis"]n_cells_per_axis / 2) + half_thickness) * self.param["l_ambda"]
            #     analytical_concentration_sand = \
            #         [y_max_sand *
            #          (special.erf((item - left) / (2 * sqrt(self.param["n_cells_per_axis"]d_coeff_out * (iteration + 1) * self.param["n_cells_per_axis"]time_total /
            #                                                 self.param["n_cells_per_axis"]number_of_iterations))) -
            #           special.erf((item - right) / (2 * sqrt(self.param["n_cells_per_axis"]d_coeff_out * (iteration + 1) * self.param["n_cells_per_axis"]time_total /
            #                                                  self.param["n_cells_per_axis"]number_of_iterations))))
            #          for item in x]
            #     ax1.set_ylim(0, y_max_sand * 2 + y_max_sand * 0.2)
            #     ax1.plot(x, analytical_concentration_sand, color='k')

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        animation = FuncAnimation(fig, animate)
        plt.show()
        # self.conn.commit()

    def plot_concentration(self, plot_separate=True, iteration=None, conc_type="atomic", analytic_sol=False):
        if iteration is None:
            iteration = self.last_i
        inward = np.zeros(self.axlim, dtype=int)
        inward_moles = np.zeros(self.axlim, dtype=int)
        inward_mass = np.zeros(self.axlim, dtype=int)

        sinward = np.zeros(self.axlim, dtype=int)
        sinward_moles = np.zeros(self.axlim, dtype=int)
        sinward_mass = np.zeros(self.axlim, dtype=int)

        outward = np.zeros(self.axlim, dtype=int)
        outward_moles = np.zeros(self.axlim, dtype=int)
        outward_mass = np.zeros(self.axlim, dtype=int)
        outward_eq_mat_moles = np.zeros(self.axlim, dtype=int)

        soutward = np.zeros(self.axlim, dtype=int)
        soutward_moles = np.zeros(self.axlim, dtype=int)
        soutward_mass = np.zeros(self.axlim, dtype=int)
        soutward_eq_mat_moles = np.zeros(self.axlim, dtype=int)

        primary_product = np.zeros(self.axlim, dtype=int)
        primary_product_moles = np.zeros(self.axlim, dtype=int)
        primary_product_moles_tc = np.zeros(self.axlim, dtype=int)
        primary_product_mass = np.zeros(self.axlim, dtype=int)
        primary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

        secondary_product = np.zeros(self.axlim, dtype=int)
        secondary_product_moles = np.zeros(self.axlim, dtype=int)
        secondary_product_moles_tc = np.zeros(self.axlim, dtype=int)
        secondary_product_mass = np.zeros(self.axlim, dtype=int)
        secondary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

        ternary_product = np.zeros(self.axlim, dtype=int)
        ternary_product_moles = np.zeros(self.axlim, dtype=int)
        ternary_product_moles_tc = np.zeros(self.axlim, dtype=int)
        ternary_product_mass = np.zeros(self.axlim, dtype=int)
        ternary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

        quaternary_product = np.zeros(self.axlim, dtype=int)
        quaternary_product_moles = np.zeros(self.axlim, dtype=int)
        quaternary_product_moles_tc = np.zeros(self.axlim, dtype=int)
        quaternary_product_mass = np.zeros(self.axlim, dtype=int)
        quaternary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

        quint_product = np.zeros(self.axlim, dtype=int)
        quint_product_moles = np.zeros(self.axlim, dtype=int)
        quint_product_moles_tc = np.zeros(self.axlim, dtype=int)
        quint_product_mass = np.zeros(self.axlim, dtype=int)
        quint_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

        if self.Config.INWARD_DIFFUSION:
            items = self._fetch_iter_table(iteration, "primary_oxidant")
            inward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
            inward_moles = inward * self.Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
            inward_mass = inward * self.Config.OXIDANTS.PRIMARY.MASS_PER_CELL

            if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                items = self._fetch_iter_table(iteration, "secondary_oxidant")
                sinward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                sinward_moles = sinward * self.Config.OXIDANTS.SECONDARY.MOLES_PER_CELL
                sinward_mass = sinward * self.Config.OXIDANTS.SECONDARY.MASS_PER_CELL

        if self.Config.OUTWARD_DIFFUSION:
            items = self._fetch_iter_table(iteration, "primary_active")
            outward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
            outward_moles = outward * self.Config.ACTIVES.PRIMARY.MOLES_PER_CELL
            outward_mass = outward * self.Config.ACTIVES.PRIMARY.MASS_PER_CELL
            outward_eq_mat_moles = outward * self.Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

            if self.Config.ACTIVES.SECONDARY_EXISTENCE:
                items = self._fetch_iter_table(iteration, "secondary_active")
                soutward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                soutward_moles = soutward * self.Config.ACTIVES.SECONDARY.MOLES_PER_CELL
                soutward_mass = soutward * self.Config.ACTIVES.SECONDARY.MASS_PER_CELL
                soutward_eq_mat_moles = soutward * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        if self.Config.COMPUTE_PRECIPITATION:
            items = self._fetch_iter_table(iteration, "primary_product")
            if np.any(items):
                primary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                primary_product_moles = primary_product * self.Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
                primary_product_moles_tc = primary_product * self.Config.PRODUCTS.PRIMARY.MOLES_PER_CELL_TC
                primary_product_mass = primary_product * self.Config.PRODUCTS.PRIMARY.MASS_PER_CELL
                primary_product_eq_mat_moles = primary_product * self.Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL *\
                                               self.Config.PRODUCTS.PRIMARY.THRESHOLD_OUTWARD

            if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                items = self._fetch_iter_table(iteration, "secondary_product")
                if np.any(items):
                    secondary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    secondary_product_moles = secondary_product * self.Config.PRODUCTS.SECONDARY.MOLES_PER_CELL
                    secondary_product_moles_tc = secondary_product * self.Config.PRODUCTS.SECONDARY.MOLES_PER_CELL_TC
                    secondary_product_mass = secondary_product * self.Config.PRODUCTS.SECONDARY.MASS_PER_CELL
                    secondary_product_eq_mat_moles = secondary_product * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL *\
                                                     self.Config.PRODUCTS.SECONDARY.THRESHOLD_OUTWARD

                items = self._fetch_iter_table(iteration, "ternary_product")
                if np.any(items):
                    ternary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    ternary_product_moles = ternary_product * self.Config.PRODUCTS.TERNARY.MOLES_PER_CELL
                    ternary_product_moles_tc = ternary_product * self.Config.PRODUCTS.TERNARY.MOLES_PER_CELL_TC
                    ternary_product_mass = ternary_product * self.Config.PRODUCTS.TERNARY.MASS_PER_CELL
                    ternary_product_eq_mat_moles = (ternary_product * ((self.Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL *
                                        self.Config.PRODUCTS.TERNARY.THRESHOLD_OUTWARD) +
                                                                       self.Config.PRODUCTS.TERNARY.MOLES_PER_CELL))

                items = self._fetch_iter_table(iteration, "quaternary_product")
                if np.any(items):
                    quaternary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    quaternary_product_moles = quaternary_product * self.Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL
                    quaternary_product_moles_tc = quaternary_product * self.Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL_TC
                    quaternary_product_mass = quaternary_product * self.Config.PRODUCTS.QUATERNARY.MASS_PER_CELL
                    quaternary_product_eq_mat_moles = (quaternary_product * ((self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL *
                                           self.Config.PRODUCTS.QUATERNARY.THRESHOLD_OUTWARD) +
                                                                             self.Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL))

                items = self._fetch_iter_table(iteration, "quint_product")
                if np.any(items):
                    quint_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    quint_product_moles = quint_product * self.Config.PRODUCTS.QUINT.MOLES_PER_CELL
                    quint_product_moles_tc = quint_product * self.Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL_TC
                    quint_product_mass = quint_product * self.Config.PRODUCTS.QUINT.MASS_PER_CELL
                    quint_product_eq_mat_moles = quint_product * self.Config.PRODUCTS.QUINT.MOLES_PER_CELL

            elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                items = self._fetch_iter_table(iteration, "secondary_product")
                if np.any(items):
                    secondary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    secondary_product_moles = secondary_product * self.Config.PRODUCTS.SECONDARY.MOLES_PER_CELL
                    secondary_product_moles_tc = secondary_product * self.Config.PRODUCTS.SECONDARY.MOLES_PER_CELL_TC
                    secondary_product_mass = secondary_product * self.Config.PRODUCTS.SECONDARY.MASS_PER_CELL
                    secondary_product_eq_mat_moles = secondary_product * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL *\
                                                      self.Config.PRODUCTS.SECONDARY.THRESHOLD_OUTWARD

        self.conn.commit()

        # n_matrix_page = (self.axlim ** 2) * self.param["product"]["primary"]["oxidation_number"]
        n_matrix_page = (self.axlim ** 2)
        # matrix = np.full(self.axlim, n_matrix_page)
        matrix_moles_per_page = n_matrix_page * self.Config.MATRIX.MOLES_PER_CELL

        matrix_moles = matrix_moles_per_page - outward_eq_mat_moles\
                       - soutward_eq_mat_moles - primary_product_eq_mat_moles - secondary_product_eq_mat_moles\
                       - ternary_product_eq_mat_moles - quaternary_product_eq_mat_moles - quint_product_eq_mat_moles

        # less_than_zero = np.where(matrix_moles < 0)[0]
        # matrix_moles[less_than_zero] = 0

        matrix_mass = matrix_moles * self.Config.MATRIX.MOLAR_MASS

        # matrix = (n_matrix_page - outward - soutward -
        #           primary_product - secondary_product - ternary_product - quaternary_product)
        # less_than_zero = np.where(matrix < 0)[0]
        # matrix[less_than_zero] = 0

        # matrix_moles = matrix * self.param["active_element"]["primary"]["eq_matrix_moles_per_cell"]
        # matrix_mass = matrix * self.param["active_element"]["primary"]["eq_matrix_mass_per_cell"]

        # x = np.linspace(0, self.param["size"] * 1000000, self.axlim)
        x = np.linspace(0, self.Config.SIZE, self.axlim)

        if conc_type.lower() == "atomic":
            conc_type_caption = "Concentration [at%]"
            whole_moles = matrix_moles +\
                          inward_moles + sinward_moles +\
                          outward_moles + soutward_moles +\
                          primary_product_moles + secondary_product_moles +\
                          ternary_product_moles + quaternary_product_moles + quint_product_moles
            whole_moles = np.maximum(whole_moles, 1e-100)

            inward = inward_moles * 100 / whole_moles
            sinward = sinward_moles * 100 / whole_moles
            outward = outward_moles * 100 / whole_moles
            soutward = soutward_moles * 100 / whole_moles

            primary_product = primary_product_moles * 100 / whole_moles
            secondary_product = secondary_product_moles * 100 / whole_moles
            ternary_product = ternary_product_moles * 100 / whole_moles
            quaternary_product = quaternary_product_moles * 100 / whole_moles
            quint_product = quint_product_moles * 100 / whole_moles

        elif conc_type.lower() == "atomic_tc":
            conc_type_caption = "Concentration [at%]"
            whole_moles = matrix_moles +\
                          inward_moles + sinward_moles +\
                          outward_moles + soutward_moles +\
                          primary_product_moles_tc + secondary_product_moles_tc +\
                          ternary_product_moles_tc + quaternary_product_moles_tc + quint_product_moles_tc
            whole_moles = np.maximum(whole_moles, 1e-100)

            inward = inward_moles * 100 / whole_moles
            sinward = sinward_moles * 100 / whole_moles
            outward = outward_moles * 100 / whole_moles
            soutward = soutward_moles * 100 / whole_moles

            primary_product = primary_product_moles_tc * 100 / whole_moles
            secondary_product = secondary_product_moles_tc * 100 / whole_moles
            ternary_product = ternary_product_moles_tc * 100 / whole_moles
            quaternary_product = quaternary_product_moles_tc * 100 / whole_moles
            quint_product = quint_product_moles_tc * 100 / whole_moles

        elif conc_type.lower() == "cells":
            conc_type_caption = "cells concentration [%]"
            n_cells_page = max(self.axlim ** 2, 1)

            inward = inward * 100 / n_cells_page
            sinward = sinward * 100 / n_cells_page
            outward = outward * 100 / n_cells_page
            soutward = soutward * 100 / n_cells_page

            primary_product = primary_product * 100 / n_cells_page
            secondary_product = secondary_product * 100 / n_cells_page
            ternary_product = ternary_product * 100 / n_cells_page
            quaternary_product = quaternary_product * 100 / n_cells_page
            quint_product = quint_product * 100 / n_cells_page

        elif conc_type.lower() == "mass":
            conc_type_caption = "Concentration [wt%]"
            whole_mass = matrix_mass +\
                         inward_mass + sinward_mass +\
                         outward_mass + soutward_mass +\
                         secondary_product_mass + primary_product_mass +\
                         ternary_product_mass + quaternary_product_mass + quint_product_mass
            whole_mass = np.maximum(whole_mass, 1e-100)

            inward = inward_mass * 100 / whole_mass
            sinward = sinward_mass * 100 / whole_mass
            outward = outward_mass * 100 / whole_mass
            soutward = soutward_mass * 100 / whole_mass

            primary_product = primary_product_mass * 100 / whole_mass
            secondary_product = secondary_product_mass * 100 / whole_mass
            ternary_product = ternary_product_mass * 100 / whole_mass
            quaternary_product = quaternary_product_mass * 100 / whole_mass
            quint_product = quint_product_mass * 100 / whole_mass

        else:
            conc_type_caption = "None"
            print("WRONG CONCENTRATION TYPE!")

        if plot_separate:
            csfont = {'fontname': FONT_NAME}
            lokal_linewidth = 0.8
            # Window 1: Oxidants (inward)
            fig1 = plt.figure()
            fig1.canvas.manager.set_window_title("Oxidants (inward diffusion)")
            ax1 = fig1.add_subplot(111)
            ax1.set_xlabel("Depth " + SIZE_UM_LABEL, **csfont)
            ax1.set_ylabel(conc_type_caption, **csfont)
            ax1.plot(x, inward, color='b', linewidth=lokal_linewidth)
            ax1.plot(x, sinward, color='deeppink', linewidth=lokal_linewidth)
            if analytic_sol:
                if conc_type == "atomic":
                    y_max = max(inward)
                elif conc_type == "cells":
                    y_max = self.Config.OXIDANTS.PRIMARY.CELLS_CONCENTRATION * 100
                elif conc_type == "mass":
                    y_max = max(inward)
                else:
                    y_max = max(inward)
                diff_in = self.Config.OXIDANTS.PRIMARY.DIFFUSION_COEFFICIENT
                analytical_concentration = y_max * special.erfc(x / (2 * sqrt(diff_in * self.Config.SIM_TIME)))
                ax1.plot(x, analytical_concentration, color='r', linewidth=1.5)
            # Window 2: Actives & precipitation
            fig2 = plt.figure()
            fig2.canvas.manager.set_window_title("Actives & precipitation (outward + products)")
            ax2 = fig2.add_subplot(111)
            ax2.set_xlabel("Depth " + SIZE_UM_LABEL, **csfont)
            ax2.set_ylabel(conc_type_caption, **csfont)
            ax2.plot(x, outward, color='g', linewidth=lokal_linewidth)
            ax2.plot(x, soutward, color='darkorange', linewidth=lokal_linewidth)
            ax2.plot(x, primary_product, color='r', linewidth=lokal_linewidth)
            ax2.plot(x, secondary_product, color='cyan', linewidth=lokal_linewidth)
            ax2.plot(x, ternary_product, color='darkgreen', linewidth=lokal_linewidth)
            ax2.plot(x, quaternary_product, color='steelblue', linewidth=lokal_linewidth)
            ax2.plot(x, quint_product, color='darkviolet', linewidth=lokal_linewidth)
            plt.show()
            plt.close('all')
            return

        fig = plt.figure()
        csfont = {'fontname': FONT_NAME}
        lokal_linewidth = 0.8
        ax = fig.add_subplot(111)
        fig.set_size_inches((10 * CM_PER_INCH, 9 * CM_PER_INCH))

        ax.plot(x, inward, color='b', linewidth=lokal_linewidth)
        ax.plot(x, outward, color='g', linewidth=lokal_linewidth)
        ax.plot(x, soutward, color='darkorange')
        ax.plot(x, primary_product, color='r', linewidth=lokal_linewidth)
        ax.plot(x, secondary_product, color='cyan')
        ax.plot(x, ternary_product, color='darkgreen')
        ax.plot(x, quaternary_product, color='steelblue')
        ax.plot(x, quint_product, color='darkviolet')

        ax.set_xlabel("Depth " + SIZE_UM_LABEL, **csfont)
        ax.set_ylabel(conc_type_caption, **csfont)
        plt.yticks(fontsize=20 * CM_PER_INCH, **csfont)
        plt.xticks(fontsize=20 * CM_PER_INCH, **csfont)

        if analytic_sol:
            if conc_type == "atomic":
                y_max = max(inward)
            elif conc_type == "cells":
                y_max = self.Config.OXIDANTS.PRIMARY.CELLS_CONCENTRATION * 100
            elif conc_type == "mass":
                y_max = max(inward)
            else:
                y_max = max(inward)
            diff_in = self.Config.OXIDANTS.PRIMARY.DIFFUSION_COEFFICIENT
            analytical_concentration = y_max * special.erfc(x / (2 * sqrt(diff_in * self.Config.SIM_TIME)))
            ax.plot(x, analytical_concentration, color='r', linewidth=1.5)

        # if analytic_sol_sand:
            #     self.c.execute("SELECT y_max_sand from description")
            #     y_max_sand = self.c.fetchone()[0] / 2
            #     self.c.execute("SELECT half_thickness from description")
            #     half_thickness = self.c.fetchone()[0]
            #     left = (int(self.param["inward_diffusion"]n_cells_per_axis / 2) - half_thickness) * self.param["inward_diffusion"]lamda
            #     right = (int(self.param["inward_diffusion"]n_cells_per_axis / 2) + half_thickness) * self.param["inward_diffusion"]lamda
            #     analytical_concentration_sand = \
            #         [y_max_sand *
            #          (special.erf(
            #              (item - left) / (2 * sqrt(self.param["inward_diffusion"]d_coeff_out * (iteration + 1) * self.param["inward_diffusion"]time_total /
            #                                        self.param["inward_diffusion"]number_of_iterations))) -
            #           special.erf(
            #               (item - right) / (2 * sqrt(self.param["inward_diffusion"]d_coeff_out * (iteration + 1) * self.param["inward_diffusion"]time_total /
            #                                          self.param["inward_diffusion"]number_of_iterations))))
            #          for item in x]
            #     ax.set_ylim(0, y_max_sand * 2 + y_max_sand * 0.2)
            #     ax.plot(x, analytical_concentration_sand, color='k')
        # plt.savefig(f'{self.db_name}_{iteration}.jpeg')

        # plt.savefig(f'W:/SIMCA/test_runs_data/{iteration}.jpeg', dpi=500)
        #
        # for x_pos, inw, otw, soutw, pp, sp, tp, qp, qip in zip(x, inward, outward, soutward, primary_product, secondary_product, ternary_product, quaternary_product, quint_product):
        #     print(x_pos * 1000000, inw, otw, soutw, pp, sp, tp, qp, qip, sep=" ")
            # print(x_pos * 1000000, " ", inw)

        # for x_pos, inw, out, ac in zip(x, inward, outward, primary_product):
        #     print(x_pos * 1000000, inw, out, ac, sep=" ")

        plt.show()

    def calculate_phase_size(self, iteration=None):
        array_3d = np.full((self.axlim, self.axlim, self.axlim), False, dtype=bool)

        if iteration is None:
            iteration = self.last_i

        if self.Config.COMPUTE_PRECIPITATION:
            items = self._fetch_iter_table(iteration, "primary_product")
            if np.any(items):
                array_3d[items[:, 0], items[:, 1], items[:, 2]] = True
                # xs_mean = []
                # xs_stdiv = []
                # xs_mean_n = []
                #
                # for x in range(self.axlim):
                #     segments_l = []
                #
                #     # mean along y
                #     for z in range(self.axlim):
                #         start_coord = 0
                #         line_started = False
                #         for y in range(self.axlim):
                #             if array_3d[z, y, x] and not line_started:
                #                 start_coord = y
                #                 line_started = True
                #                 continue
                #
                #             if not array_3d[z, y, x] and line_started:
                #                 new_segment_l = y - start_coord
                #
                #                 segments_l.append(new_segment_l)
                #                 line_started = False
                #
                #     # mean along z
                #     for y in range(self.axlim):
                #         start_coord = 0
                #         line_started = False
                #         for z in range(self.axlim):
                #             if array_3d[z, y, x] and not line_started:
                #                 start_coord = z
                #                 line_started = True
                #                 continue
                #
                #             if not array_3d[z, y, x] and line_started:
                #                 new_segment_l = z - start_coord
                #
                #                 segments_l.append(new_segment_l)
                #                 line_started = False
                #
                #     # stats for x plane
                #     xs_mean.append(np.mean(segments_l))
                #     xs_stdiv.append(np.std(segments_l))
                #
                # for mean, stdiv in zip(xs_mean, xs_stdiv):
                #     print(mean, " ", stdiv)

                # Label connected components (clusters)
                labeled_array, num_features = ndimage.label(array_3d)

                # Initialize a dictionary to store cluster statistics for each X position
                cluster_stats_by_x = {}

                # Iterate over slices along the X-axis
                for x in range(array_3d.shape[0]):
                    x_slice = labeled_array[:, :, x]

                    # Count cluster sizes in this slice
                    cluster_sizes = np.bincount(x_slice.ravel())

                    # Remove clusters with label 0 (background)
                    cluster_sizes = cluster_sizes[1:]

                    # Store cluster statistics for this X position
                    cluster_stats_by_x[x] = {
                        'num_clusters': len(cluster_sizes),
                        'cluster_sizes': cluster_sizes
                    }

                for x_pos in range(self.axlim):

                    clusters = np.array(cluster_stats_by_x[x_pos]["cluster_sizes"])
                    clusters = clusters[np.nonzero(clusters)]

                    mean = np.mean(clusters)
                    stdiv = np.std(clusters)

                    nz_len = len(clusters)
                    print(x_pos, nz_len, mean, stdiv, sep=" ")

    def plot_h(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        self.c.execute("SELECT * from precip_front_p")
        items = np.array(self.c.fetchall())
        if np.any(items):
            sqr_time = items[:, 0]
            position = items[:, 1]
            ax1.scatter(sqr_time, position, s=10, color='r')
        else:
            return print("No Data to plot primary precipitation front!")

        if self.Config.ACTIVES.SECONDARY_EXISTENCE:
            self.c.execute("SELECT * from precip_front_s")
            items = np.array(self.c.fetchall())
            if np.any(items):
                sqr_time_s = items[:, 0]
                position_s = items[:, 1]
                ax1.scatter(sqr_time_s, position_s, s=10, color='cyan')
            else:
                return print("No Data to plot secondary precipitation front!")
        plt.show()

    def plot_plane0_product_tracking(self):
        try:
            self.c.execute(
                """SELECT iteration, product, jmatpro_conc, existing_conc, diff_conc
                   FROM product_plane0_tracking
                   ORDER BY iteration, product"""
            )
            rows = self.c.fetchall()
        except sql.OperationalError:
            return print("No product_plane0_tracking table in this database!")

        if not rows:
            return print("No plane-0 product tracking data to plot!")

        data = pd.DataFrame(
            rows,
            columns=["iteration", "product", "jmatpro_conc", "existing_conc", "diff_conc"],
        )

        fig, ax = plt.subplots()
        for product in sorted(data["product"].unique()):
            prod_data = data[data["product"] == product].sort_values("iteration")
            ax.plot(
                prod_data["iteration"].to_numpy(),
                prod_data["jmatpro_conc"].to_numpy(),
                label=f"{product} jmatpro",
            )
            ax.plot(
                prod_data["iteration"].to_numpy(),
                prod_data["existing_conc"].to_numpy(),
                linestyle="--",
                label=f"{product} existing",
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Concentration")
        ax.set_title("Plane-0 concentration tracking")
        ax.grid(True, alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.show()


def plot_kinetics(data_to_plot, with_kinetic=False, file_path=None):
    """Plot kinetics from CSV. If file_path is None, a file dialog is shown."""
    if file_path is None:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
    if not file_path:
        return
    data = pd.read_csv(file_path, sep=" ", header=None)
    plt.figure(figsize=(10, 6))
    x_values = data.iloc[:, 0]

    some_data = []
    # some_data.append(x_values * 0.04239 * (1/3600))

    for rows in data_to_plot:
        x = x_values.copy()
        index = 2 * rows + 1
        y_values = data.iloc[:, index]
        z_ind = np.where(y_values == 0)[0]
        y_values = np.delete(y_values, z_ind)
        x = np.delete(x, z_ind)
        # plt.plot(x_values, y_values, label=f'Layer - {rows}', s=1)
        x *= 0.04239 * (1/3600)
        plt.plot(x, y_values, label=f'Layer - {rows}')
        some_data.append([x, y_values * 100])

        if with_kinetic:
            x = x_values.copy()
            y_values_soll = data.iloc[:, index + 1]
            z_ind = np.where(y_values_soll == 0)[0]
            y_values_soll = np.delete(y_values_soll, z_ind)
            x = np.delete(x, z_ind)
            # plt.plot(x_values, y_values_soll, label=f'Layer - {rows} kinetic', s=1)
            # x *= 0.00035
            plt.plot(x, y_values_soll, label=f'Layer - {rows} kinetic')

    # x_values *= 0.04239 * (1/3600)
    # for item in x_values:
    #     print(item)
    # for item in some_data:
    #     for dat in item:
    #         print(dat)
    #
    #     print(":::::::::::::::::::")
    # Find the maximum length of x and y_values
    # max_len = max(
    #     max(len(x) if isinstance(x, (list, np.ndarray)) else 0,
    #         len(y) if isinstance(y, (list, np.ndarray)) else 0)
    #     for x, y in some_data
    # )
    #
    # # Prepare the data for columns, filling shorter lists/arrays with None
    # columns = []
    # for x, y_values in some_data:
    #     x = x if isinstance(x, (list, np.ndarray)) else []
    #     y_values = y_values if isinstance(y_values, (list, np.ndarray)) else []
    #
    #     x_extended = list(x) + [None] * (max_len - len(x))
    #     y_extended = list(y_values) + [None] * (max_len - len(y_values))
    #
    #     columns.append(x_extended)
    #     columns.append(y_extended)
    #
    # # Transpose columns for saving as rows
    # transposed = list(zip(*columns))
    #
    # # Save to text file
    # output_file = "output_data.txt"
    # with open(output_file, "w") as f:
    #     for row in transposed:
    #         f.write("\t".join(str(value) if value is not None else "" for value in row) + "\n")
    #
    # print(f"Data saved to {output_file}")


    plt.xlabel("Time [sec]")
    plt.ylabel('Concentration')
    # plt.legend()
    plt.show()


def plot_kinetics_mult_comb(data_to_plot, number_of_dbs, with_kinetic=False, file_paths=None):
    """Plot kinetics from multiple CSVs. If file_paths is None, file dialogs are shown."""
    plt.figure(figsize=(10, 6))
    if file_paths is None:
        file_paths = []
        for _ in range(number_of_dbs):
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename()
            if not path:
                return
            file_paths.append(path)
    for file_path in file_paths:
        data = pd.read_csv(file_path, sep=" ", header=None)
        x_values = data.iloc[:, 0]

        for rows in data_to_plot:
            x = x_values.copy()
            index = 2 * rows + 1
            y_values = data.iloc[:, index]
            z_ind = np.where(y_values == 0)[0]
            y_values = np.delete(y_values, z_ind)
            x = np.delete(x, z_ind)
            # plt.plot(x_values, y_values, label=f'Layer - {rows}', s=1)
            # x *= 0.00035
            plt.plot(x, y_values, label=f'Layer - {rows}')

            if with_kinetic:
                x = x_values.copy()
                y_values_soll = data.iloc[:, index + 1]
                z_ind = np.where(y_values_soll == 0)[0]
                y_values_soll = np.delete(y_values_soll, z_ind)
                x = np.delete(x, z_ind)
                # plt.plot(x_values, y_values_soll, label=f'Layer - {rows} kinetic', s=1)
                # x *= 0.00035
                plt.plot(x, y_values_soll, label=f'Layer - {rows} kinetic')

    plt.xlabel("Time [sec]")
    plt.ylabel('Concentration')
    # plt.legend()
    plt.show()

