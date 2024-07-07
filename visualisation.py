import matplotlib.pyplot as plt
import sqlite3 as sql
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import special
from math import *
import numpy as np
import utils
from scipy import ndimage
import pickle
from configuration import Config
from configuration import update_class_from_dict
import pandas as pd
import tkinter as tk
from tkinter import filedialog


class Visualisation:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sql.connect(self.db_name)
        self.c = self.conn.cursor()
        self.Config = None
        self.axlim = None
        self.shape = None
        self.last_i = None
        self.oxid_numb = None
        self.utils = utils.Utils()
        self.generate_param_from_db()
        self.cell_size = 40
        self.linewidth = 0.3
        self.alpha = 1
        self.cm = {1: np.array([255, 200, 200])/255.0,
                   2: np.array([255, 75, 75])/255.0,
                   3: np.array([220, 0, 0])/255.0,
                   4: np.array([120, 0, 0])/255.0}

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

        self.utils.print_static_params(Config)
        self.c.execute("SELECT last_i from time_parameters")
        self.last_i = self.c.fetchone()[0]
        self.compute_elapsed_time()

        self.axlim = self.Config.N_CELLS_PER_AXIS
        self.shape = (self.axlim, self.axlim, self.axlim)
        self.oxid_numb = self.Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER

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

        def animate_sep(iteration):
            ax_inward.cla()
            ax_sinward.cla()
            ax_outward.cla()
            ax_soutward.cla()
            ax_precip.cla()
            ax_sprecip.cla()
            ax_tprecip.cla()
            ax_qtprecip.cla()
            if self.Config.INWARD_DIFFUSION:
                self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ax_inward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='b',
                                      s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_sinward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='deeppink',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.Config.OUTWARD_DIFFUSION:
                self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ax_outward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='g', s=3)
                if self.Config.ACTIVES.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_soutward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkorange',
                                            s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.Config.COMPUTE_PRECIPITATION:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ax_precip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='r',
                                      s=self.cell_size * (72. / fig.dpi) ** 2)

                if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_sprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)

                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_tprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkorange',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)

                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_qtprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='steelblue',
                                            s=self.cell_size * (72. / fig.dpi) ** 2)

                elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_sprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
            ax_inward.set_xlim3d(0, self.axlim)
            ax_inward.set_ylim3d(0, self.axlim)
            ax_inward.set_zlim3d(0, self.axlim)
            ax_sinward.set_xlim3d(0, self.axlim)
            ax_sinward.set_ylim3d(0, self.axlim)
            ax_sinward.set_zlim3d(0, self.axlim)
            ax_outward.set_xlim3d(0, self.axlim)
            ax_outward.set_ylim3d(0, self.axlim)
            ax_outward.set_zlim3d(0, self.axlim)
            ax_soutward.set_xlim3d(0, self.axlim)
            ax_soutward.set_ylim3d(0, self.axlim)
            ax_soutward.set_zlim3d(0, self.axlim)
            ax_precip.set_xlim3d(0, self.axlim)
            ax_precip.set_ylim3d(0, self.axlim)
            ax_precip.set_zlim3d(0, self.axlim)
            ax_sprecip.set_xlim3d(0, self.axlim)
            ax_sprecip.set_ylim3d(0, self.axlim)
            ax_sprecip.set_zlim3d(0, self.axlim)
            ax_tprecip.set_xlim3d(0, self.axlim)
            ax_tprecip.set_ylim3d(0, self.axlim)
            ax_tprecip.set_zlim3d(0, self.axlim)
            ax_qtprecip.set_xlim3d(0, self.axlim)
            ax_qtprecip.set_ylim3d(0, self.axlim)
            ax_qtprecip.set_zlim3d(0, self.axlim)
            if const_cam_pos:
                azim = -70
                elev = 30
                dist = 8
                ax_inward.azim = azim
                ax_inward.elev = elev
                ax_inward.dist = dist
                ax_sinward.azim = azim
                ax_sinward.elev = elev
                ax_sinward.dist = dist
                ax_outward.azim = azim
                ax_outward.elev = elev
                ax_outward.dist = dist
                ax_soutward.azim = azim
                ax_soutward.elev = elev
                ax_soutward.dist = dist
                ax_sprecip.azim = azim
                ax_sprecip.elev = elev
                ax_sprecip.dist = dist
                ax_sprecip.azim = azim
                ax_sprecip.elev = elev
                ax_sprecip.dist = dist

        def animate(iteration):
            ax_all.cla()
            ax_all.dist = 4
            # if self.Config.INWARD_DIFFUSION:
            #     self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
            #     items = np.array(self.c.fetchall())
            #     if np.any(items):
            #         ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='b',
            #                        s=self.cell_size * (72. / fig.dpi) ** 2)
            #
            #     if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
            #         self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
            #         items = np.array(self.c.fetchall())
            #         if np.any(items):
            #             ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='deeppink',
            #                            s=self.cell_size * (72. / fig.dpi) ** 2)
            # if self.Config.OUTWARD_DIFFUSION:
            #     self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
            #     items = np.array(self.c.fetchall())
            #     if np.any(items):
            #         ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='g',
            #                        s=self.cell_size * (72. / fig.dpi) ** 2)
            #     if self.Config.ACTIVES.SECONDARY_EXISTENCE:
            #         self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
            #         items = np.array(self.c.fetchall())
            #         if np.any(items):
            #             ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkorange',
            #                            s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.Config.COMPUTE_PRECIPITATION:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    # items = items.transpose()
                    # data = np.zeros(self.shape, dtype=bool)
                    # data[items[0], items[1], items[2]] = True
                    # ax_all.voxels(data, facecolors="r")
                    # plt.savefig(f'W:/SIMCA/test_runs_data/{iteration}.jpeg')
                    # ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='r',
                    #                s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth)
                    #
                    counts = np.unique(np.ravel_multi_index(items.transpose(), self.shape), return_counts=True)
                    dec = np.array(np.unravel_index(counts[0], self.shape), dtype=np.short).transpose()
                    counts = np.array(counts[1], dtype=np.ubyte)
                    full_ind = np.where(counts == self.oxid_numb)[0]

                    fulls = dec[full_ind]
                    not_fulls = np.delete(dec, full_ind, axis=0)

                    ax_all.scatter(fulls[:, 2], fulls[:, 1], fulls[:, 0], marker=',', color="darkred",
                                   s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black',
                                   linewidth=self.linewidth,
                                   alpha=self.alpha)

                    ax_all.scatter(not_fulls[:, 2], not_fulls[:, 1], not_fulls[:, 0], marker=',', color='r',
                                   s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black',
                                   linewidth=self.linewidth,
                                   alpha=self.alpha)

                if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)

                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkorange',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)

                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='steelblue',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)

                elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)

            ax_all.set_xlim3d(0, self.axlim)
            ax_all.set_ylim3d(0, self.axlim)
            ax_all.set_zlim3d(0, self.axlim)

            if const_cam_pos:
                ax_all.azim = -45
                ax_all.elev = 22
                ax_all.dist = 7.5

        fig = plt.figure()
        # fig.set_size_inches(18.5, 10.5)
        if animate_separate:
            ax_inward = fig.add_subplot(341, projection='3d')
            ax_sinward = fig.add_subplot(345, projection='3d')
            ax_outward = fig.add_subplot(342, projection='3d')
            ax_soutward = fig.add_subplot(346, projection='3d')

            ax_precip = fig.add_subplot(349, projection='3d')
            ax_sprecip = fig.add_subplot(3, 4, 10, projection='3d')
            ax_tprecip = fig.add_subplot(3, 4, 11, projection='3d')
            ax_qtprecip = fig.add_subplot(3, 4, 12, projection='3d')
            animation = FuncAnimation(fig, animate_sep)

        else:
            ax_all = fig.add_subplot(111, projection='3d')
            animation = FuncAnimation(fig, animate)
        plt.show()
        # plt.savefig(f'C:/test_runs_data/{"_"}.jpeg')

    def plot_3d(self, plot_separate=False, iteration=None, const_cam_pos=False):
        if iteration is None:
            iteration = self.last_i
        fig = plt.figure()
        new_axlim = self.axlim
        rescale_factor = new_axlim / self.axlim
        if plot_separate:
            ax_inward = fig.add_subplot(341, projection='3d')
            ax_sinward = fig.add_subplot(345, projection='3d')
            ax_outward = fig.add_subplot(342, projection='3d')
            ax_soutward = fig.add_subplot(346, projection='3d')

            ax_precip = fig.add_subplot(349, projection='3d')
            ax_sprecip = fig.add_subplot(3,4,10, projection='3d')
            ax_tprecip = fig.add_subplot(3,4,11, projection='3d')
            ax_qtprecip = fig.add_subplot(3,4,12, projection='3d')

            if self.Config.INWARD_DIFFUSION:
                self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ax_inward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='b',
                                      s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
                                   alpha=self.alpha)

                if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_sinward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='deeppink',
                                           s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
                                   alpha=self.alpha)
            if self.Config.OUTWARD_DIFFUSION:
                self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ax_outward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='g',
                                       s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
                                   alpha=self.alpha)
                if self.Config.ACTIVES.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_soutward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkorange',
                                            s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
                                   alpha=self.alpha)
            if self.Config.COMPUTE_PRECIPITATION:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ax_precip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='r',
                                      s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
                                   alpha=self.alpha)

                if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_sprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                                           s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
                                   alpha=self.alpha)

                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_tprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkgreen',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)

                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_qtprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='steelblue',
                                            s=self.cell_size * (72. / fig.dpi) ** 2)

                elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_sprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='saddlebrown',
                                           s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
                                   alpha=self.alpha)

            ax_inward.set_xlim3d(0, self.axlim)
            ax_inward.set_ylim3d(0, self.axlim)
            ax_inward.set_zlim3d(0, self.axlim)
            ax_sinward.set_xlim3d(0, self.axlim)
            ax_sinward.set_ylim3d(0, self.axlim)
            ax_sinward.set_zlim3d(0, self.axlim)
            ax_outward.set_xlim3d(0, self.axlim)
            ax_outward.set_ylim3d(0, self.axlim)
            ax_outward.set_zlim3d(0, self.axlim)
            ax_soutward.set_xlim3d(0, self.axlim)
            ax_soutward.set_ylim3d(0, self.axlim)
            ax_soutward.set_zlim3d(0, self.axlim)
            ax_precip.set_xlim3d(0, self.axlim)
            ax_precip.set_ylim3d(0, self.axlim)
            ax_precip.set_zlim3d(0, self.axlim)
            ax_sprecip.set_xlim3d(0, self.axlim)
            ax_sprecip.set_ylim3d(0, self.axlim)
            ax_sprecip.set_zlim3d(0, self.axlim)
            ax_tprecip.set_xlim3d(0, self.axlim)
            ax_tprecip.set_ylim3d(0, self.axlim)
            ax_tprecip.set_zlim3d(0, self.axlim)
            ax_qtprecip.set_xlim3d(0, self.axlim)
            ax_qtprecip.set_ylim3d(0, self.axlim)
            ax_qtprecip.set_zlim3d(0, self.axlim)

            if const_cam_pos:
                azim = -92
                elev = 0
                dist = 8
                ax_inward.azim = azim
                ax_inward.elev = elev
                ax_inward.dist = dist
                ax_sinward.azim = azim
                ax_sinward.elev = elev
                ax_sinward.dist = dist
                ax_outward.azim = azim
                ax_outward.elev = elev
                ax_outward.dist = dist
                ax_soutward.azim = azim
                ax_soutward.elev = elev
                ax_soutward.dist = dist
                ax_precip.azim = azim
                ax_precip.elev = elev
                ax_precip.dist = dist
                ax_sprecip.azim = azim
                ax_sprecip.elev = elev
                ax_sprecip.dist = dist
                ax_tprecip.azim = azim
                ax_tprecip.elev = elev
                ax_tprecip.dist = dist
                ax_qtprecip.azim = azim
                ax_qtprecip.elev = elev
                ax_qtprecip.dist = dist
        else:

            ax_all = fig.add_subplot(111, projection='3d')

            # # Define the grid for the plane
            # x = np.linspace(0, new_axlim, 100)
            # y = np.linspace(0, new_axlim, 100)
            # X, Y = np.meshgrid(x, y)
            # # Define the Z coordinates for the plane
            # Z = np.full(X.shape,
            #             new_axlim / 2)  # This sets Z = 2 for the entire plane, making it parallel to the YX axis
            # # Plot the plane
            # ax_all.plot_surface(X, Y, Z, color='r', alpha=0.5, zorder=0)  # Set alpha to a value between 0 and 1 for transparency

            # if self.Config.INWARD_DIFFUSION:
            #     self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
            #     items = np.array(self.c.fetchall())
            #     if np.any(items):
            #         items = items * rescale_factor
            #         ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='b',
            #                        s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
            #                        alpha=self.alpha)

            #     if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
            #         self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
            #         items = np.array(self.c.fetchall())
            #         if np.any(items):
            #             ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='deeppink',
            #                            s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
            #                        alpha=self.alpha)
            # if self.Config.OUTWARD_DIFFUSION:
            #     self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
            #     items = np.array(self.c.fetchall())
            #     if np.any(items):
            #         items = items * rescale_factor
            #         ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='g',
            #                        s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
            #                        alpha=self.alpha)
            #     if self.Config.ACTIVES.SECONDARY_EXISTENCE:
            #         self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
            #         items = np.array(self.c.fetchall())
            #         if np.any(items):
            #             items = items * rescale_factor
            #             ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='gold',
            #                            s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
            #                        alpha=self.alpha)

            if self.Config.COMPUTE_PRECIPITATION:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    # items = items * rescale_factor
                    # ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color="darkred",
                    #                s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black',
                    #                linewidth=self.linewidth,
                    #                alpha=self.alpha)

                    counts = np.unique(np.ravel_multi_index(items.transpose(), self.shape), return_counts=True)
                    dec = np.array(np.unravel_index(counts[0], self.shape), dtype=np.short).transpose()
                    counts = np.array(counts[1], dtype=np.ubyte)

                    # cube_size = 1
                    # some_max_numb = 4
                    #
                    # # Create and plot a cube for each center coordinate
                    # for center, transparency in zip(dec, counts):
                    #     # Map transparency to the alpha value (1 is fully opaque, 0 is fully transparent)
                    #     alpha = 1 - (transparency - 1) / (some_max_numb - 1)
                    #
                    #     # Define the vertices of the cube based on the center and size
                    #     r = cube_size / 2
                    #     vertices = np.array([
                    #         [center[2] - r, center[1] - r, center[0] - r],
                    #         [center[2] + r, center[1] - r, center[0] - r],
                    #         [center[2] + r, center[1] + r, center[0] - r],
                    #         [center[2] - r, center[1] + r, center[0] - r],
                    #         [center[2] - r, center[1] - r, center[0] + r],
                    #         [center[2] + r, center[1] - r, center[0] + r],
                    #         [center[2] + r, center[1] + r, center[0] + r],
                    #         [center[2] - r, center[1] + r, center[0] + r]
                    #     ])
                    #
                    #     # Define the faces of the cube
                    #     faces = [
                    #         [vertices[j] for j in [0, 1, 2, 3]],
                    #         [vertices[j] for j in [4, 5, 6, 7]],
                    #         [vertices[j] for j in [0, 3, 7, 4]],
                    #         [vertices[j] for j in [1, 2, 6, 5]],
                    #         [vertices[j] for j in [0, 1, 5, 4]],
                    #         [vertices[j] for j in [2, 3, 7, 6]]
                    #     ]
                    #
                    #     # Create a Poly3DCollection for the cube with opaque faces
                    #     cube = Poly3DCollection(faces, alpha=alpha, linewidths=0.1, edgecolors='k', facecolors='r')
                    #     ax_all.add_collection3d(cube)

                    # for grade in range(1, 5):
                    #     grade_ind = np.where(counts == grade)[0]
                    #     ax_all.scatter(dec[grade_ind, 2], dec[grade_ind, 1], dec[grade_ind, 0], marker=',',
                    #                    color=self.cm[grade], s=self.cell_size * (72. / fig.dpi) ** 2)

                    full_ind = np.where(counts == self.oxid_numb)[0]

                    fulls = dec[full_ind]
                    not_fulls = np.delete(dec, full_ind, axis=0)

                    ax_all.scatter(fulls[:, 2], fulls[:, 1], fulls[:, 0], marker=',', color="darkred",
                                   s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
                                   alpha=self.alpha)

                    ax_all.scatter(not_fulls[:, 2], not_fulls[:, 1], not_fulls[:, 0], marker=',', color='r',
                                   s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
                                   alpha=self.alpha)

                if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)

                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkgreen',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)

                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='steelblue',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)

                if self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        items = items * rescale_factor
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='tomato',
                                       s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth,
                                   alpha=self.alpha)

            ax_all.set_xlim3d(0, self.axlim * rescale_factor)
            ax_all.set_ylim3d(0, self.axlim * rescale_factor)
            ax_all.set_zlim3d(0, self.axlim * rescale_factor)
            if const_cam_pos:
                ax_all.azim = -62
                ax_all.elev = 17
                ax_all.dist = 2

        cm = 1 / 2.54  # centimeters in inches
        #
        # # fig.set_size_inches((12*cm, 12*cm))
        # # plt.savefig(f'C:/test_runs_data/{iteration}.jpeg')
        # # plt.savefig(f"//juno/homes/user/aseregin/Desktop/simuls/{iteration}.jpeg")
        #
        # csfont = {'fontname': 'Times New Roman'}
        # # # # Rescale the axis values
        # ticks = np.arange(0, new_axlim + 1, 500)
        # ax_all.set_xticks(ticks)
        # ax_all.set_yticks(ticks)
        # ax_all.set_zticks(ticks)
        #
        # # Set font properties for the ticks
        # f_size = 40
        # ax_all.tick_params(axis='x', labelsize=f_size * cm, labelcolor='black', pad=10)
        # ax_all.tick_params(axis='y', labelsize=f_size * cm, labelcolor='black', pad=10)
        # ax_all.tick_params(axis='z', labelsize=f_size * cm, labelcolor='black', pad=10)
        #
        # # Get the tick labels and set font properties
        # for tick in ax_all.get_xticklabels():
        #     tick.set_fontname('Times New Roman')
        # for tick in ax_all.get_yticklabels():
        #     tick.set_fontname('Times New Roman')
        # for tick in ax_all.get_zticklabels():
        #     tick.set_fontname('Times New Roman')
        #
        # ax_all.set_xlabel("X [µm]", **csfont, fontsize=f_size*cm, labelpad=17)
        # ax_all.set_ylabel("Y [µm]", **csfont, fontsize=f_size*cm, labelpad=17)
        # ax_all.set_zlabel("Z [µm]", **csfont, fontsize=f_size*cm, labelpad=17)

        # fig.set_size_inches((40 * cm, 40 * cm))
        plt.show()
        # plt.savefig(f'C:/test_runs_data/{iteration}.jpeg', dpi=300)
        plt.close()

    def plot_2d(self, plot_separate=False, iteration=None, slice_pos=None):
        if iteration is None:
            iteration = self.last_i
        if slice_pos is None:
            slice_pos = int(self.axlim / 2)
        fig = plt.figure()
        if plot_separate:
            ax_inward = fig.add_subplot(341)
            ax_sinward = fig.add_subplot(345)
            ax_outward = fig.add_subplot(342)
            ax_soutward = fig.add_subplot(346)

            ax_precip = fig.add_subplot(349)
            ax_sprecip = fig.add_subplot(3, 4, 10)
            ax_tprecip = fig.add_subplot(3, 4, 11)
            ax_qtprecip = fig.add_subplot(3, 4, 12)

            if self.Config.INWARD_DIFFUSION:
                self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_inward.scatter(items[ind, 2], items[ind, 1], marker=',', color='b',
                                      s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_sinward.scatter(items[ind, 2], items[ind, 1], marker=',', color='deeppink',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.Config.OUTWARD_DIFFUSION:
                self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_outward.scatter(items[ind, 2], items[ind, 1], marker=',', color='g',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.Config.ACTIVES.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_soutward.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkorange',
                                            s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.Config.COMPUTE_PRECIPITATION:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_precip.scatter(items[ind, 2], items[ind, 1], marker=',', color='r',
                                      s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_sprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_tprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkgreen',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_qtprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='steelblue',
                                            s=self.cell_size * (72. / fig.dpi) ** 2)
                elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_sprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)

            ax_inward.set_xlim(0, self.axlim)
            ax_inward.set_ylim(0, self.axlim)
            ax_sinward.set_xlim(0, self.axlim)
            ax_sinward.set_ylim(0, self.axlim)
            ax_outward.set_xlim(0, self.axlim)
            ax_outward.set_ylim(0, self.axlim)
            ax_soutward.set_xlim(0, self.axlim)
            ax_soutward.set_ylim(0, self.axlim)
            ax_precip.set_xlim(0, self.axlim)
            ax_precip.set_ylim(0, self.axlim)
            ax_sprecip.set_xlim(0, self.axlim)
            ax_sprecip.set_ylim(0, self.axlim)
            ax_tprecip.set_xlim(0, self.axlim)
            ax_tprecip.set_ylim(0, self.axlim)
            ax_qtprecip.set_xlim(0, self.axlim)
            ax_qtprecip.set_ylim(0, self.axlim)
        else:
            ax_all = fig.add_subplot(111)
            ax_all.set_facecolor('gainsboro')
            if self.Config.INWARD_DIFFUSION:
                self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='b',
                                   s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth)
                if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='deeppink',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.Config.OUTWARD_DIFFUSION:
                self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='g',
                                   s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth)
                if self.Config.ACTIVES.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkorange',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
            #
            if self.Config.COMPUTE_PRECIPITATION:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    # ind = np.where(items[:, 2] == slice_pos)

                    items = np.array(items[ind]).transpose()

                    counts = np.unique(np.ravel_multi_index(items, self.shape), return_counts=True)
                    dec = np.array(np.unravel_index(counts[0], self.shape), dtype=np.short).transpose()
                    counts = np.array(counts[1], dtype=np.ubyte)

                    # for grade in range(1, 5):
                    #     grade_ind = np.where(counts == grade)[0]
                    #     ax_all.scatter(dec[grade_ind, 2], dec[grade_ind, 1], marker=',',
                    #                    color=self.cm[grade], s=self.cell_size * (72. / fig.dpi) ** 2)

                    full_ind = np.where(counts == self.oxid_numb)[0]
                    #
                    fulls = dec[full_ind]
                    not_fulls = np.delete(dec, full_ind, axis=0)

                    ax_all.scatter(fulls[:, 2], fulls[:, 1], marker=',', color='darkred',
                                   s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth)

                    ax_all.scatter(not_fulls[:, 2], not_fulls[:, 1], marker=',', color='r',
                                   s=self.cell_size * (72. / fig.dpi) ** 2, edgecolors='black', linewidth=self.linewidth)

                    # ax_all.scatter(fulls[:, 1], fulls[:, 0], marker=',', color='r',
                    #                s=self.cell_size * (72. / fig.dpi) ** 2)
                    #
                    # ax_all.scatter(not_fulls[:, 1], not_fulls[:, 0], marker=',', color='r',
                    #                s=self.cell_size * (72. / fig.dpi) ** 2, )

                    # ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='r',
                    #                s=self.cell_size * (72. / fig.dpi) ** 2)

                if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkgreen',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='steelblue',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
            ax_all.set_xlim(0, self.axlim)
            ax_all.set_ylim(0, self.axlim)
        self.conn.commit()
        # plt.savefig(f'W:/SIMCA/test_runs_data/{slice_pos}.jpeg')
        # plt.savefig(f"//juno/homes/user/aseregin/Desktop/Neuer Ordner/{slice_pos}.jpeg")
        plt.show()

    def animate_2d(self, plot_separate=False, slice_pos=None):
        if not self.Config.SAVE_WHOLE:
            return print("No Data To Animate!")

        def animate_sep(iteration):
            if self.Config.INWARD_DIFFUSION:
                ax_inward.cla()
                self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_inward.scatter(items[ind, 2], items[ind, 1], marker=',', color='b',
                                      s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    ax_sinward.cla()
                    self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_sinward.scatter(items[ind, 2], items[ind, 1], marker=',', color='deeppink',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.Config.OUTWARD_DIFFUSION:
                ax_outward.cla()
                self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_outward.scatter(items[ind, 2], items[ind, 1], marker=',', color='g',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.Config.ACTIVES.SECONDARY_EXISTENCE:
                    ax_soutward.cla()
                    self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_soutward.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkorange',
                                            s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.Config.COMPUTE_PRECIPITATION:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_precip.scatter(items[ind, 2], items[ind, 1], marker=',', color='r',
                                      s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_sprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_tprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkgreen',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_qtprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='steelblue',
                                            s=self.cell_size * (72. / fig.dpi) ** 2)
                elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_sprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
            ax_inward.set_xlim(0, self.axlim)
            ax_inward.set_ylim(0, self.axlim)
            ax_sinward.set_xlim(0, self.axlim)
            ax_sinward.set_ylim(0, self.axlim)
            ax_outward.set_xlim(0, self.axlim)
            ax_outward.set_ylim(0, self.axlim)
            ax_soutward.set_xlim(0, self.axlim)
            ax_soutward.set_ylim(0, self.axlim)
            ax_precip.set_xlim(0, self.axlim)
            ax_precip.set_ylim(0, self.axlim)
            ax_sprecip.set_xlim(0, self.axlim)
            ax_sprecip.set_ylim(0, self.axlim)

        def animate(iteration):
            ax_all.cla()
            if self.Config.INWARD_DIFFUSION:
                self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='b',
                                   s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='deeppink',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.Config.OUTWARD_DIFFUSION:
                self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='g',
                                   s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.Config.ACTIVES.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkorange',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.Config.COMPUTE_PRECIPITATION:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='r',
                                   s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkgreen',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='steelblue',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
            ax_all.set_xlim(0, self.axlim)
            ax_all.set_ylim(0, self.axlim)

        if slice_pos is None:
            slice_pos = int(self.axlim / 2)
        fig = plt.figure()
        if plot_separate:
            ax_inward = fig.add_subplot(341)
            ax_sinward = fig.add_subplot(345)
            ax_outward = fig.add_subplot(342)
            ax_soutward = fig.add_subplot(346)

            ax_precip = fig.add_subplot(349)
            ax_sprecip = fig.add_subplot(3, 4, 10)
            ax_tprecip = fig.add_subplot(3, 4, 11)
            ax_qtprecip = fig.add_subplot(3, 4, 12)
            animation = FuncAnimation(fig, animate_sep)
        else:
            ax_all = fig.add_subplot(111)
            animation = FuncAnimation(fig, animate)
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
                self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                inward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                inward_moles = inward * self.Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
                inward_mass = inward * self.Config.OXIDANTS.PRIMARY.MASS_PER_CELL

                if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    sinward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    sinward_moles = sinward * self.Config.OXIDANTS.SECONDARY.MOLES_PER_CELL
                    sinward_mass = sinward * self.Config.OXIDANTS.SECONDARY.MASS_PER_CELL

            if self.Config.OUTWARD_DIFFUSION:
                self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                outward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                outward_moles = outward * self.Config.ACTIVES.PRIMARY.MOLES_PER_CELL
                outward_mass = outward * self.Config.ACTIVES.PRIMARY.MASS_PER_CELL
                outward_eq_mat_moles = outward * self.Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

                if self.Config.ACTIVES.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    soutward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    soutward_moles = soutward * self.Config.ACTIVES.SECONDARY.MOLES_PER_CELL
                    soutward_mass = soutward * self.Config.ACTIVES.SECONDARY.MASS_PER_CELL
                    soutward_eq_mat_moles = soutward * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

            if self.Config.COMPUTE_PRECIPITATION:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    primary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    primary_product_moles = primary_product * self.Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
                    primary_product_mass = primary_product * self.Config.PRODUCTS.PRIMARY.MASS_PER_CELL
                    primary_product_eq_mat_moles = primary_product * self.Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

                if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    secondary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    secondary_product_moles = secondary_product * self.Config.PRODUCTS.SECONDARY.MOLES_PER_CELL
                    secondary_product_mass = secondary_product * self.Config.PRODUCTS.SECONDARY.MASS_PER_CELL
                    secondary_product_eq_mat_moles = secondary_product * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    ternary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    ternary_product_moles = ternary_product * self.Config.PRODUCTS.TERNARY.MOLES_PER_CELL
                    ternary_product_mass = ternary_product * self.Config.PRODUCTS.TERNARY.MASS_PER_CELL
                    ternary_product_eq_mat_moles = ternary_product * self.Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    quaternary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    quaternary_product_moles = quaternary_product * self.Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL
                    quaternary_product_mass = quaternary_product * self.Config.PRODUCTS.QUATERNARY.MASS_PER_CELL
                    quaternary_product_eq_mat_moles = quaternary_product * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

                elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        secondary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                        secondary_product_moles = secondary_product * self.Config.PRODUCTS.SECONDARY.MOLES_PER_CELL
                        secondary_product_mass = secondary_product * self.Config.PRODUCTS.SECONDARY.MASS_PER_CELL
                        secondary_product_eq_mat_moles = primary_product * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

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

                inward = inward_moles * 100 / whole_moles
                sinward = sinward_moles * 100 / whole_moles
                outward = outward_moles * 100 / whole_moles
                soutward = soutward_moles * 100 / whole_moles

                primary_product = primary_product_moles * 100 / whole_moles
                secondary_product = secondary_product_moles * 100 / whole_moles
                ternary_product = ternary_product_moles * 100 / whole_moles
                quaternary_product = quaternary_product_moles * 100 / whole_moles

            elif conc_type.lower() == "cells":
                n_cells_page = self.axlim ** 2
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

        if self.Config.INWARD_DIFFUSION:
            self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
            items = np.array(self.c.fetchall())
            inward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
            inward_moles = inward * self.Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
            inward_mass = inward * self.Config.OXIDANTS.PRIMARY.MASS_PER_CELL

            if self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                sinward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                sinward_moles = sinward * self.Config.OXIDANTS.SECONDARY.MOLES_PER_CELL
                sinward_mass = sinward * self.Config.OXIDANTS.SECONDARY.MASS_PER_CELL

        if self.Config.OUTWARD_DIFFUSION:
            self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
            items = np.array(self.c.fetchall())
            outward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
            outward_moles = outward * self.Config.ACTIVES.PRIMARY.MOLES_PER_CELL
            outward_mass = outward * self.Config.ACTIVES.PRIMARY.MASS_PER_CELL
            outward_eq_mat_moles = outward * self.Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

            if self.Config.ACTIVES.SECONDARY_EXISTENCE:
                self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                soutward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                soutward_moles = soutward * self.Config.ACTIVES.SECONDARY.MOLES_PER_CELL
                soutward_mass = soutward * self.Config.ACTIVES.SECONDARY.MASS_PER_CELL
                soutward_eq_mat_moles = soutward * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        if self.Config.COMPUTE_PRECIPITATION:
            self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
            items = np.array(self.c.fetchall())
            if np.any(items):
                primary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                primary_product_moles = primary_product * self.Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
                # primary_product_moles_tc = primary_product * self.Config.PRODUCTS.PRIMARY.MOLES_PER_CELL_TC
                primary_product_mass = primary_product * self.Config.PRODUCTS.PRIMARY.MASS_PER_CELL
                primary_product_eq_mat_moles = primary_product * self.Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

            if self.Config.ACTIVES.SECONDARY_EXISTENCE and self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                secondary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                secondary_product_moles = secondary_product * self.Config.PRODUCTS.SECONDARY.MOLES_PER_CELL
                # secondary_product_moles_tc = secondary_product * self.Config.PRODUCTS.SECONDARY.MOLES_PER_CELL_TC
                secondary_product_mass = secondary_product * self.Config.PRODUCTS.SECONDARY.MASS_PER_CELL
                secondary_product_eq_mat_moles = secondary_product * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

                self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                ternary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                ternary_product_moles = ternary_product * self.Config.PRODUCTS.TERNARY.MOLES_PER_CELL
                # ternary_product_moles_tc = ternary_product * self.Config.PRODUCTS.TERNARY.MOLES_PER_CELL_TC
                ternary_product_mass = ternary_product * self.Config.PRODUCTS.TERNARY.MASS_PER_CELL
                ternary_product_eq_mat_moles = ternary_product * self.Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

                self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                quaternary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                # quaternary_product_moles_tc = quaternary_product * self.Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL
                quaternary_product_moles = quaternary_product * self.Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL
                quaternary_product_mass = quaternary_product * self.Config.PRODUCTS.QUATERNARY.MASS_PER_CELL
                quaternary_product_eq_mat_moles = quaternary_product * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

            elif self.Config.ACTIVES.SECONDARY_EXISTENCE and not self.Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    secondary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    secondary_product_moles = secondary_product * self.Config.PRODUCTS.SECONDARY.MOLES_PER_CELL
                    secondary_product_mass = secondary_product * self.Config.PRODUCTS.SECONDARY.MASS_PER_CELL
                    secondary_product_eq_mat_moles = primary_product * self.Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        self.conn.commit()

        # n_matrix_page = (self.axlim ** 2) * self.param["product"]["primary"]["oxidation_number"]
        n_matrix_page = (self.axlim ** 2)
        matrix = np.full(self.axlim, n_matrix_page)

        matrix_moles = matrix * self.Config.MATRIX.MOLES_PER_CELL - outward_eq_mat_moles\
                       - soutward_eq_mat_moles - primary_product_eq_mat_moles - secondary_product_eq_mat_moles\
                       - ternary_product_eq_mat_moles - quaternary_product_eq_mat_moles

        less_than_zero = np.where(matrix_moles < 0)[0]
        matrix_moles[less_than_zero] = 0

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
                          ternary_product_moles + quaternary_product_moles

            inward = inward_moles * 100 / whole_moles
            sinward = sinward_moles * 100 / whole_moles
            outward = outward_moles * 100 / whole_moles
            soutward = soutward_moles * 100 / whole_moles

            primary_product = primary_product_moles * 100 / whole_moles
            secondary_product = secondary_product_moles * 100 / whole_moles
            ternary_product = ternary_product_moles * 100 / whole_moles
            quaternary_product = quaternary_product_moles * 100 / whole_moles

        elif conc_type.lower() == "atomic_tc":
            conc_type_caption = "Concentration [at%]"
            whole_moles = matrix_moles +\
                          inward_moles + sinward_moles +\
                          outward_moles + soutward_moles +\
                          primary_product_moles_tc + secondary_product_moles_tc +\
                          ternary_product_moles_tc + quaternary_product_moles_tc

            inward = inward_moles * 100 / whole_moles
            sinward = sinward_moles * 100 / whole_moles
            outward = outward_moles * 100 / whole_moles
            soutward = soutward_moles * 100 / whole_moles

            primary_product = primary_product_moles_tc * 100 / whole_moles
            secondary_product = secondary_product_moles_tc * 100 / whole_moles
            ternary_product = ternary_product_moles_tc * 100 / whole_moles
            quaternary_product = quaternary_product_moles_tc * 100 / whole_moles

        elif conc_type.lower() == "cells":
            conc_type_caption = "cells concentration [%]"
            n_cells_page = (self.axlim ** 2) * self.Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER

            # DELETE!!!
            n_cells_page = (self.axlim ** 2)

            inward = inward * 100 / n_cells_page
            sinward = sinward * 100 / n_cells_page
            outward = outward * 100 / n_cells_page
            soutward = soutward * 100 / n_cells_page

            primary_product = primary_product * 100 / n_cells_page
            secondary_product = secondary_product * 100 / n_cells_page
            ternary_product = ternary_product * 100 / n_cells_page
            quaternary_product = quaternary_product * 100 / n_cells_page

        elif conc_type.lower() == "mass":
            conc_type_caption = "Concentration [wt%]"
            whole_mass = matrix_mass +\
                         inward_mass + sinward_mass +\
                         outward_mass + soutward_mass +\
                         secondary_product_mass + primary_product_mass +\
                         ternary_product_mass + quaternary_product_mass

            inward = inward_mass * 100 / whole_mass
            sinward = sinward_mass * 100 / whole_mass
            outward = outward_mass * 100 / whole_mass
            soutward = soutward_mass * 100 / whole_mass

            primary_product = primary_product_mass * 100 / whole_mass
            secondary_product = secondary_product_mass * 100 / whole_mass
            ternary_product = ternary_product_mass * 100 / whole_mass
            quaternary_product = quaternary_product_mass * 100 / whole_mass

        else:
            conc_type_caption = "None"
            print("WRONG CONCENTRATION TYPE!")

        fig = plt.figure()
        if plot_separate:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.plot(x, inward, color='b')
            ax1.plot(x, sinward, color='deeppink')

            ax2.plot(x, outward, color='g')
            ax2.plot(x, soutward, color='darkorange')

            ax2.plot(x, primary_product, color='r')
            ax2.plot(x, secondary_product, color='cyan')
            ax2.plot(x, ternary_product, color='darkgreen')
            ax2.plot(x, quaternary_product, color='steelblue')

            if analytic_sol:
                if conc_type == "atomic":
                    y_max = max(inward)
                    y_max_out = self.Config.ACTIVES.PRIMARY.ATOMIC_CONCENTRATION * 100

                elif conc_type == "cells":
                    y_max = self.Config.OXIDANTS.PRIMARY.CELLS_CONCENTRATION * 100
                    y_max_out = self.Config.ACTIVES.PRIMARY.CELLS_CONCENTRATION * 100

                elif conc_type == "mass":
                    y_max = max(inward)
                    y_max_out = self.Config.ACTIVES.PRIMARY.MASS_CONCENTRATION * 100

                diff_in = self.Config.OXIDANTS.PRIMARY.DIFFUSION_COEFFICIENT
                diff_out = self.Config.ACTIVES.PRIMARY.DIFFUSION_COEFFICIENT

                analytical_concentration = y_max * special.erfc(x / (2 * sqrt(diff_in * self.onfig.SIM_TIME)))
                analytical_concentration_out = (y_max_out / 2) * (1 - special.erf((- x + 0.0005) / (2 * sqrt(
                    diff_out * (iteration + 1) * self.Config.SIM_TIME / self.Config.N_ITERATIONS))))

                # ax1.set_ylim(0, y_max + y_max * 0.2)
                # ax2.set_ylim(0, y_max_out + y_max_out * 0.2)
                ax2.plot(x, analytical_concentration_out, color='r', linewidth=1.5)
                ax1.plot(x, analytical_concentration, color='r', linewidth=1.5)
        else:
            csfont = {'fontname': 'Times New Roman'}
            lokal_linewidth = 0.8

            cm = 1 / 2.54  # centimeters in inches

            ax = fig.add_subplot(111)
            fig.set_size_inches((10 * cm, 9 * cm))

            ax.plot(x, inward, color='b', linewidth=lokal_linewidth)
            # ax.plot(x, sinward, color='deeppink')
            ax.plot(x, outward, color='g', linewidth=lokal_linewidth)
            # ax.plot(x, soutward, color='darkorange')
            ax.plot(x, primary_product, color='r', linewidth=lokal_linewidth)
            # ax.plot(x, secondary_product, color='cyan')
            # ax.plot(x, ternary_product, color='darkgreen')
            # ax.plot(x, quaternary_product, color='steelblue')

            ax.set_xlabel("Depth [µm]", **csfont)
            ax.set_ylabel(conc_type_caption, **csfont)
            # plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
            # plt.xticks([0, 50, 100, 150, 200, 250, 300])
            plt.yticks(fontsize=20 * cm, **csfont)
            plt.xticks(fontsize=20 * cm, **csfont)

            # ax.plot(x, outward,  color='g')
            # ax.plot(x, precipitations,  color='r')

            if analytic_sol:
                y_max = self.Config.OXIDANTS.PRIMARY.CELLS_CONCENTRATION * 100
                analytical_concentration = y_max * special.erfc(x / (2 * sqrt(self.Config.OXIDANTS.PRIMARY.DIFFUSION_COEFFICIENT *
                                                                              self.Config.SIM_TIME)))
                ax.set_ylim(0, y_max + y_max * 0.2)
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
        # for x_pos, inw, outw, prod in zip(x, inward, outward, primary_product):
            # print(x_pos * 1000000, " ", inw, " ", outw,  " ", prod)
            # print(x_pos * 1000000, " ", inw)

        # for inw in inward:
        #     print(inw)

        plt.show()

    def calculate_phase_size(self, iteration=None):
        array_3d = np.full((self.axlim, self.axlim, self.axlim), False, dtype=bool)

        if iteration is None:
            iteration = self.last_i

        if self.Config.COMPUTE_PRECIPITATION:
            self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
            items = np.array(self.c.fetchall())
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
                    nz_len = len(clusters)
                    print(x_pos, nz_len, mean, sep=" ")

    def plot_h(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        self.conn = sql.connect(self.db_name)
        self.c = self.conn.cursor()

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

def plot_kinetics(data_to_plot, with_kinetic=False):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    data = pd.read_csv(file_path, sep=" ", header=None)
    plt.figure(figsize=(10, 6))
    x_values = data.iloc[:, 0]

    for rows in data_to_plot:
        index = 2 * rows + 1
        y_values = data.iloc[:, index]
        # plt.plot(x_values, y_values, label=f'Layer - {rows}', s=1)
        plt.plot(x_values, y_values, label=f'Layer - {rows}')

        if with_kinetic:
            y_values_soll = data.iloc[:, index + 1]
            # plt.plot(x_values, y_values_soll, label=f'Layer - {rows} kinetic', s=1)
            plt.plot(x_values, y_values_soll, label=f'Layer - {rows} kinetic')

    plt.xlabel("Time [sec]")
    plt.ylabel('Concentration')
    plt.legend()
    plt.show()
