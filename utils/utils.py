from .physical_data import *
import sys
import numpy as np
import time
import datetime
import pprint
from configuration import Config
import math
from types import SimpleNamespace


class Utils:
    def __init__(self):
        self.param = 0
        self.n_cells_per_axis = Config.N_CELLS_PER_AXIS
        self.neigh_range = Config.NEIGH_RANGE

        self.ind_decompose = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1],   # 5 flat
             [1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],      # 9  corners
             [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1],  # 13
             [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],  # 19 side corners
             [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=np.byte)

        self.ind_decompose_no_flat = np.array(
            [[1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],
             [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1],
             [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],
             [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=np.byte)

        self.ind_decompose_flat = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.byte)

        self.ind_decompose_flat_z = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 0]], dtype=np.byte)

        self.ind_formation1 = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 0],
             [1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],
             [-1, 1, -1], [-1, 1, 1], [-1, -1, -1],
             [-1, -1, 1], [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],
             [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=np.byte)

        if self.neigh_range > 1:
            # self.ind_formation = self.generate_neigh_indexes()
            # self.ind_formation = self.generate_neigh_indexes_squashed()
            self.ind_formation = self.generate_neigh_indexes_flat()
        else:
            self.ind_formation = self.generate_neigh_indexes_flat()
            # self.ind_formation = np.array(
            #     [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 0],
            #       [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]], dtype=np.byte)

        self.ind_formation_noz = np.array(np.delete(self.ind_formation1, 13, 0), dtype=np.byte)

        self.interface_neigh = {(0, 0, 1): [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
                                (0, 0, -1): [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1]],
                                (0, 1, 0): [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0]],
                                (0, -1, 0): [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0]],
                                (1, 0, 0): [[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 0, 0]],
                                (-1, 0, 0): [[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [-1, 0, 0]]}

        self.interface_neigh_adj = {(0, 0, 1): [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 0]],
                                (0, 0, -1): [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 0]],
                                (0, 1, 0): [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0]],
                                (0, -1, 0): [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0]],
                                (1, 0, 0): [[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0]],
                                (-1, 0, 0): [[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0]]}

    def generate_param(self):
        initial_input_snapshot = None
        if Config.SAVE_POST_PROCESSED_INPUT:
            initial_input_snapshot = self._snapshot_static_params(Config)

        Config.GENERATED_VALUES.TAU = Config.SIM_TIME / Config.N_ITERATIONS
        Config.GENERATED_VALUES.LAMBDA = Config.SIZE / Config.N_CELLS_PER_AXIS

        Config.GENERATED_VALUES.KINETIC_KONST = (Config.ZETTA_FINAL-Config.ZETTA_ZERO) / (Config.SIM_TIME ** 0.5)

        self._prepare_element_lists_dynamic()
        self._calc_active_data_dynamic()
        self._calc_oxidant_data_dynamic()
        self.calc_product_data()
        self._calc_product_diffusion_probability_maps()
        self._build_runtime_element_settings()
        self._calc_initial_conc_and_moles_dynamic()

        time_stamp = int(time.time())
        # Config.GENERATED_VALUES.DB_ID = str(int(time_stamp + random.randint(1, 1000000)))
        Config.GENERATED_VALUES.DB_ID = str(time_stamp)
        Config.GENERATED_VALUES.DB_PATH = Config.SAVE_PATH + Config.GENERATED_VALUES.DB_ID + '.db'
        Config.GENERATED_VALUES.DATE_OF_CREATION = str(datetime.datetime.fromtimestamp(time_stamp))
        print("DB_PATH: ", Config.GENERATED_VALUES.DB_PATH)

        if Config.SAVE_POST_PROCESSED_INPUT:
            path = Config.SAVE_PATH + str(int(time.time())) + '_config.txt'
            with open(path, 'w', encoding='utf-8') as file:
                if initial_input_snapshot is not None:
                    file.write("INITIAL_CONFIG:\n")
                    file.write(self._format_pretty_snapshot(initial_input_snapshot))
                    file.write("\n\nPOST_PROCESSED_INPUT:\n")
                self.print_static_params_to_file(Config, file)

    @staticmethod
    def _prepare_element_lists_dynamic():
        # Normalize user input and filter "None" entries.
        norm_actives = []
        for item in Config.ACTIVES:
            elem = str(item.get("element", "")).strip()
            item["element"] = elem
            item["mass_concentration"] = item.get("mass_concentration", 0.0)
            item["cells_concentration"] = item.get("cells_concentration", 0.0)
            item["conc_precision"] = item.get("conc_precision", "rand")
            item["space_fill"] = item.get("space_fill", "full")
            item["diffusion_max_per_cell"] = item.get("diffusion_max_per_cell", 10)
            norm_actives.append(item)
        Config.ACTIVES = norm_actives

        norm_oxidants = []
        for item in Config.OXIDANTS:
            elem = str(item.get("element", "")).strip()
            item["element"] = elem
            item["cells_concentration"] = item.get("cells_concentration", 0.0)
            item["diffusion_max_per_cell"] = item.get("diffusion_max_per_cell", 3)
            item["diffusion_condition_gb"] = str(item.get("diffusion_condition_gb", item.get("diffusion_condition", "")))
            norm_oxidants.append(item)
        Config.OXIDANTS = norm_oxidants

    def _calc_active_data_dynamic(self):
        matrix_elem = Config.MATRIX.ELEMENT
        Config.MATRIX.DENSITY = DENSITY[matrix_elem]
        Config.MATRIX.MOLAR_MASS = MOLAR_MASS[matrix_elem]
        cell_volume = Config.GENERATED_VALUES.LAMBDA ** 3
        Config.MATRIX.MOLES_PER_CELL = Config.MATRIX.DENSITY * cell_volume / Config.MATRIX.MOLAR_MASS
        Config.MATRIX.MASS_PER_CELL = Config.MATRIX.MOLES_PER_CELL * Config.MATRIX.MOLAR_MASS

        for active in Config.ACTIVES:
            elem = active["element"]
            active["DENSITY"] = DENSITY[elem]
            active["MOLAR_MASS"] = MOLAR_MASS[elem]
            active["DIFFUSION_COEFFICIENT"] = get_diff_coeff(Config.TEMPERATURE, active["diffusion_condition"])
            active["PROBABILITIES"] = self.calc_prob(active["DIFFUSION_COEFFICIENT"], stridden=True)
            active["DIFFUSION_COEFFICIENT_IN_PRODUCT"] = self._resolve_in_product_diff_coeff(
                active,
                active["DIFFUSION_COEFFICIENT"],
            )
            active["PROBABILITIES_IN_PRODUCT"] = self.calc_prob(
                active["DIFFUSION_COEFFICIENT_IN_PRODUCT"],
                stridden=True,
            )

        mass_sum = sum(float(a["mass_concentration"]) for a in Config.ACTIVES)
        if mass_sum > 1.0:
            raise ValueError(f"Sum of active mass concentrations must be <= 1.0, got {mass_sum}")

        denom = (1.0 - mass_sum) / Config.MATRIX.MOLAR_MASS
        for active in Config.ACTIVES:
            denom += float(active["mass_concentration"]) / float(active["MOLAR_MASS"])
        denom = max(denom, 1e-30)

        denom_moles = 1.0
        for active in Config.ACTIVES:
            atomic_c = (float(active["mass_concentration"]) / float(active["MOLAR_MASS"])) / denom
            active["ATOMIC_CONCENTRATION"] = atomic_c
            t_val = float(active["MOLAR_MASS"]) * float(Config.MATRIX.DENSITY) / (
                float(active["DENSITY"]) * float(Config.MATRIX.MOLAR_MASS)
            )
            active["T"] = t_val
            active["n_ELEM"] = 1.0 - t_val
            denom_moles += atomic_c * (t_val - 1.0)
        denom_moles = max(denom_moles, 1e-30)

        for active in Config.ACTIVES:
            min_cells = float(active["ATOMIC_CONCENTRATION"]) * float(active["T"]) / denom_moles
            min_cells = max(0.0, min_cells)

            if Config.FULL_CELLS:
                cells_conc = min_cells
                active["cells_concentration"] = cells_conc
            else:
                cells_conc = float(active["cells_concentration"])
                if cells_conc < min_cells:
                    raise ValueError(
                        f"Cells concentration for outward element '{active['element']}' must be >= {min_cells}"
                    )

            if cells_conc > 0.0:
                moles_per_cell = (
                    float(active["ATOMIC_CONCENTRATION"]) * float(Config.MATRIX.MOLES_PER_CELL)
                ) / (cells_conc * denom_moles)
            else:
                moles_per_cell = 0.0
            active["MOLES_PER_CELL"] = moles_per_cell
            active["MASS_PER_CELL"] = moles_per_cell * float(active["MOLAR_MASS"])
            active["EQ_MATRIX_MOLES_PER_CELL"] = moles_per_cell * float(active["T"])
            active["EQ_MATRIX_MASS_PER_CELL"] = active["EQ_MATRIX_MOLES_PER_CELL"] * float(Config.MATRIX.MOLAR_MASS)
            active["N_PER_PAGE"] = round(cells_conc * Config.N_CELLS_PER_AXIS ** 2)

    def _calc_oxidant_data_dynamic(self):
        if len(Config.ACTIVES) == 0:
            raise ValueError("ACTIVES list is empty.")
        ref_active_moles = float(Config.ACTIVES[0]["MOLES_PER_CELL"])

        first_product = Config.PRODUCTS[0] if isinstance(Config.PRODUCTS, list) and len(Config.PRODUCTS) > 0 else {}
        stoich = first_product.get("stoich", {})
        outward = str(first_product.get("outward_element", "")).strip()
        inward = str(first_product.get("inward_element", "")).strip()
        thr_out = float(first_product.get("threshold_outward", 1.0))
        thr_in = float(first_product.get("threshold_inward", 1.0))
        nu_out = float(stoich.get(outward, 0.0))
        nu_in = float(stoich.get(inward, 0.0))
        ratio_in_to_out = (nu_in / max(nu_out, 1e-30)) * (thr_out / max(thr_in, 1e-30))
        ref_oxidant_moles = ref_active_moles * ratio_in_to_out

        for oxidant in Config.OXIDANTS:
            elem = oxidant["element"]
            oxidant["DENSITY"] = DENSITY[elem]
            oxidant["MOLAR_MASS"] = MOLAR_MASS[elem]
            oxidant["DIFFUSION_COEFFICIENT"] = get_diff_coeff(Config.TEMPERATURE, oxidant["diffusion_condition"])
            oxidant["DIFFUSION_COEFFICIENT_GB"] = get_diff_coeff(
                Config.TEMPERATURE, oxidant.get("diffusion_condition_gb", oxidant["diffusion_condition"])
            )
            oxidant["PROBABILITIES"] = self.calc_prob(oxidant["DIFFUSION_COEFFICIENT"])
            oxidant["DIFFUSION_COEFFICIENT_IN_PRODUCT"] = self._resolve_in_product_diff_coeff(
                oxidant,
                oxidant["DIFFUSION_COEFFICIENT"],
            )
            oxidant["PROBABILITIES_IN_PRODUCT"] = self.calc_prob(
                oxidant["DIFFUSION_COEFFICIENT_IN_PRODUCT"]
            )
            oxidant["PROBABILITIES_2D"] = self.calc_p0_2d(oxidant["DIFFUSION_COEFFICIENT_GB"])
            oxidant["PROBABILITIES_SCALE"] = self.calc_prob(oxidant["DIFFUSION_COEFFICIENT"] * 10 ** -2)
            oxidant["PROBABILITIES_INTERFACE"] = self.calc_prob(oxidant["DIFFUSION_COEFFICIENT"] * 10 ** 3)
            oxidant["N_PER_PAGE"] = round(float(oxidant["cells_concentration"]) * Config.N_CELLS_PER_AXIS ** 2)
            oxidant["MOLES_PER_CELL"] = ref_oxidant_moles
            oxidant["MASS_PER_CELL"] = ref_oxidant_moles * float(oxidant["MOLAR_MASS"])
            oxidant["ATOMIC_FRACTION"] = float(oxidant["atomic_fraction"])
            oxidant["K_CONST"] = oxidant["ATOMIC_FRACTION"] / (ref_oxidant_moles * (1 - oxidant["ATOMIC_FRACTION"]))

    @staticmethod
    def _resolve_in_product_diff_coeff(species_cfg, base_coeff):
        coeff_override = species_cfg.get("diffusion_coefficient_in_product", None)
        cond_override = species_cfg.get("diffusion_condition_in_product", None)
        if coeff_override is not None and str(coeff_override).strip() != "":
            return float(coeff_override)
        cond_text = str(cond_override).strip() if cond_override is not None else ""
        if cond_text not in ("", "None", "none"):
            return float(get_diff_coeff(Config.TEMPERATURE, str(cond_override)))
        return float(base_coeff)

    @staticmethod
    def _resolve_product_phase_override(raw_override):
        if raw_override is None:
            return None, None
        if isinstance(raw_override, dict):
            coeff_override = raw_override.get("coefficient", None)
            cond_override = raw_override.get("condition", None)
            return coeff_override, cond_override
        if isinstance(raw_override, (int, float, np.floating)):
            return float(raw_override), None
        if isinstance(raw_override, str):
            return None, raw_override
        return None, None

    def _calc_product_diffusion_probability_maps(self):
        products = getattr(Config, "PRODUCTS", None)
        product_defs = []
        if isinstance(products, (list, tuple)):
            for idx, prod in enumerate(products):
                priority = int(prod.get("priority", 0))
                product_defs.append((priority, idx, prod))
            product_defs.sort(key=lambda item: (item[0], item[1]))

        species_entries = [(cfg, True) for cfg in Config.ACTIVES] + [(cfg, False) for cfg in Config.OXIDANTS]
        for species_cfg, use_stridden in species_entries:
            elem = str(species_cfg.get("element", "")).strip()
            coeff_by_phase = {}
            probs_by_phase = {}
            for phase_offset, (_, _, prod) in enumerate(product_defs, start=1):
                raw_map = prod.get("diffusion_in_product", None)
                raw_override = None
                if isinstance(raw_map, dict):
                    if elem in raw_map:
                        raw_override = raw_map[elem]
                    elif "*" in raw_map:
                        raw_override = raw_map["*"]
                elif raw_map is not None:
                    raw_override = raw_map

                coeff_override, cond_override = self._resolve_product_phase_override(raw_override)
                if coeff_override is not None and str(coeff_override).strip() != "":
                    coeff_val = float(coeff_override)
                elif cond_override is not None and str(cond_override).strip() not in ("", "None", "none"):
                    coeff_val = float(get_diff_coeff(Config.TEMPERATURE, str(cond_override)))
                else:
                    coeff_val = float(species_cfg["DIFFUSION_COEFFICIENT_IN_PRODUCT"])

                coeff_by_phase[int(phase_offset)] = coeff_val
                probs_by_phase[int(phase_offset)] = self.calc_prob(coeff_val, stridden=use_stridden)

            species_cfg["PRODUCT_DIFFUSION_COEFFICIENT_BY_PHASE"] = coeff_by_phase
            species_cfg["PRODUCT_PROBABILITIES_BY_PHASE"] = probs_by_phase

    @staticmethod
    def _build_runtime_element_settings():
        def to_runtime_obj(d):
            ns = SimpleNamespace()
            for k, v in d.items():
                setattr(ns, k.upper(), v)
            return ns
        Config.ACTIVES_RUNTIME = [to_runtime_obj(a) for a in Config.ACTIVES]
        Config.OXIDANTS_RUNTIME = [to_runtime_obj(o) for o in Config.OXIDANTS]

    @staticmethod
    def _calc_initial_conc_and_moles_dynamic():
        # Keep only key totals that are consumed in runtime diagnostics.
        inward_moles = sum(int(o["N_PER_PAGE"]) * float(o["MOLES_PER_CELL"]) for o in Config.OXIDANTS)
        outward_moles = sum(int(a["N_PER_PAGE"]) * float(a["MOLES_PER_CELL"]) for a in Config.ACTIVES)
        matrix_moles = float(Config.N_CELLS_PER_AXIS ** 2) * float(Config.MATRIX.MOLES_PER_CELL)
        whole_moles = matrix_moles + inward_moles + outward_moles
        Config.GENERATED_VALUES.inward_moles = inward_moles
        Config.GENERATED_VALUES.outward_moles = outward_moles
        Config.GENERATED_VALUES.matrix_moles = matrix_moles
        Config.GENERATED_VALUES.whole_moles = whole_moles
        Config.GENERATED_VALUES.max_gamma_min_one = 0 if Config.SOL_PROD == 0 else ((inward_moles ** 3) * (outward_moles ** 2)) / Config.SOL_PROD - 1

    @staticmethod
    def calc_product_data():
        products = getattr(Config, "PRODUCTS", None)

        act_cfg_by_elem = {}
        ox_cfg_by_elem = {}
        for cfg in Config.ACTIVES:
            elem = str(cfg.get("element", "None"))
            if elem and elem.lower() != "none":
                act_cfg_by_elem[elem] = cfg
        for cfg in Config.OXIDANTS:
            elem = str(cfg.get("element", "None"))
            if elem and elem.lower() != "none":
                ox_cfg_by_elem[elem] = cfg

        for prod in products:
            stoich = {str(k): int(v) for k, v in prod.get("stoich", {}).items()}
            outward_raw = prod.get("outward_element", "")
            inward_raw = prod.get("inward_element", "")
            outward = "" if outward_raw is None else str(outward_raw).strip()
            inward = "" if inward_raw is None else str(inward_raw).strip()
            matrix_elem = str(getattr(Config.MATRIX, "ELEMENT", ""))

            prod["components"] = list(stoich.keys())
            thr_out = int(prod.get("threshold_outward", 0))
            thr_in = int(prod.get("threshold_inward", 0))
            prod["THRESHOLD_OUTWARD"] = thr_out
            prod["THRESHOLD_INWARD"] = thr_in

            in_cfg = ox_cfg_by_elem.get(inward)
            if in_cfg is None:
                raise ValueError(
                    f"Product '{prod.get('element', '<unknown>')}' inward_element '{inward}' is not configured in OXIDANTS."
                )

            ref_cfg = None
            if outward:
                ref_cfg = act_cfg_by_elem.get(outward)
                if ref_cfg is None:
                    raise ValueError(
                        f"Product '{prod.get('element', '<unknown>')}' outward_element '{outward}' is not configured in ACTIVES."
                    )
            elif thr_out != 0:
                raise ValueError(
                    f"Product '{prod.get('element', '<unknown>')}' has no outward_element, so threshold_outward must be 0."
                )

            # Product mass and moles per cell-event are both threshold-driven.
            out_mass = float(ref_cfg["MASS_PER_CELL"]) * float(thr_out) if ref_cfg is not None else 0.0
            in_mass = float(in_cfg["MASS_PER_CELL"]) * float(thr_in)

            out_moles = float(ref_cfg["MOLES_PER_CELL"]) * float(thr_out) if ref_cfg is not None else 0.0
            in_moles = float(in_cfg["MOLES_PER_CELL"]) * float(thr_in)

            # Optional matrix contribution for products whose stoich includes matrix element (e.g., spinels).
            matrix_moles = 0.0
            nu_matrix = float(stoich.get(matrix_elem, 0.0))
            if nu_matrix > 0.0:
                nu_out = float(stoich.get(outward, 0.0))
                nu_in = float(stoich.get(inward, 0.0))
                matrix_candidates = []
                if nu_out > 0.0:
                    matrix_candidates.append(out_moles * (nu_matrix / nu_out))
                if nu_in > 0.0:
                    matrix_candidates.append(in_moles * (nu_matrix / nu_in))
                if len(matrix_candidates) > 0:
                    matrix_moles = float(sum(matrix_candidates) / len(matrix_candidates))
            matrix_mass = matrix_moles * float(getattr(Config.MATRIX, "MOLAR_MASS", 0.0))

            prod["MATRIX_MOLES_PER_CELL"] = matrix_moles
            prod["MATRIX_MASS_PER_CELL"] = matrix_mass
            prod["MASS_PER_CELL"] = out_mass + in_mass + matrix_mass
            prod["MOLES_PER_CELL"] = out_moles + in_moles + matrix_moles
            prod["CONSTITUTION"] = "+".join(prod["components"])

            t_val = float(ref_cfg.get("T", 0.0)) if ref_cfg is not None else 0.0
            thr_out = max(1, int(prod["THRESHOLD_OUTWARD"]))
            if ref_cfg is not None and t_val > 0.0 and float(ref_cfg["MOLES_PER_CELL"]) > 0.0:
                ox_num = math.floor(
                    (float(Config.MATRIX.MOLES_PER_CELL) / (float(prod["stoich"].get(Config.MATRIX.ELEMENT, 0) * matrix_moles + ref_cfg["MOLES_PER_CELL"]) * t_val * thr_out))
                )
            elif ref_cfg is None and float(prod["MOLES_PER_CELL"]) > 0.0:
                # No outward reactant case: cap by matrix moles available per matrix cell-event.
                ox_num = math.floor(float(Config.MATRIX.MOLES_PER_CELL) / float(prod["stoich"][Config.MATRIX.ELEMENT] * matrix_moles))

            prod["OXIDATION_NUMBER"] = max(1, int(ox_num))

    @staticmethod
    def calc_prob(diff_coeff, stridden=False):
        if not stridden:
            coeff = 6 * (Config.GENERATED_VALUES.TAU * diff_coeff) / (Config.GENERATED_VALUES.LAMBDA ** 2)
        else:
            new_tau = Config.SIM_TIME / (Config.N_ITERATIONS / Config.STRIDE)
            coeff = 6 * (new_tau * diff_coeff) / (Config.GENERATED_VALUES.LAMBDA ** 2)
        c = (1 - coeff) / (1 + coeff)
        if -(c + 1) / 8 > (c - 1) / 8:
            t = -(c + 1) / 16
        else:
            t = (c - 1) / 16
        p = -2 * t
        p3 = (1 / (coeff + 1)) - 2 * p
        p0 = 1 - 4 * p - p3

        # sum0 = 4 * p + p0 + p3
        # s0diff_coeff = ((Config.GENERATED_VALUES.LAMBDA ** 2) / new_tau) * (
        #             (1 + p0 - p3) / (6 * (1 - p0 + p3)))
        #
        # c = 6 * new_tau * diff_coeff / (6 * new_tau * diff_coeff + (Config.GENERATED_VALUES.LAMBDA ** 2))
        # if c < 1/2:
        #     t = -c/4
        #     print("Range: [", -c/2, " : 0]")
        # else:
        #     t = (c-1)/4
        #     print("Range: [", (c-1) / 2, " : 0]")
        # np = -t
        # np3 = 2 * t - c + 1
        # np0 = 2 * t + c
        # sum = 4*np + np0 + np3
        # s1diff_coeff = ((Config.GENERATED_VALUES.LAMBDA ** 2)/new_tau) * ((1 + np0 - np3)/(6 * (1 - np0 + np3)))

        return [p, p3, p0]

    @staticmethod
    def calc_p0_2d(diff_coeff):
        coeff = 4 * (Config.GENERATED_VALUES.TAU * diff_coeff) / (Config.GENERATED_VALUES.LAMBDA ** 2)
        coeff_p = coeff / (1 + coeff)
        if coeff_p < 1:
            r_bound = 1
        else:
            r_bound = 1 / coeff_p
        l_bound = 0
        if 1 - (1/(2*coeff_p)) < 2 - (1/coeff_p):
            if 2 - (1/coeff_p) > 0:
                l_bound = 2 - (1/coeff_p)

        elif 1 - (1/(2*coeff_p)) > 2 - (1/coeff_p):
            if 1 - (1/(2*coeff_p)) > 0:
                l_bound = 1 - (1/(2*coeff_p))
        t = l_bound + (r_bound - l_bound) / 2
        # p = coeff_p * (1 - t)
        p0 = t * coeff_p
        # p3 = 1 + coeff_p * (t - 2)
        return p0

    @staticmethod
    def calc_prob_manually(p, diff_coeff):
        coeff = 6 * (Config.GENERATED_VALUES.TAU * diff_coeff) / (Config.GENERATED_VALUES.LAMBDA ** 2)
        p3 = (1 - 2 * p * (1 + coeff)) / (1 + coeff)
        p0 = 1 - 4 * p - p3
        return {"p": p, "p3": p3, "p0": p0}

   

    def generate_neigh_indexes(self):
        neigh_range = self.neigh_range
        size = 3 + (neigh_range - 1) * 2
        neigh_shape = (size, size, size)
        temp = np.ones(neigh_shape, dtype=int)
        coord = np.array(np.nonzero(temp))
        coord -= neigh_range
        coord = coord.transpose()
        return np.array(coord, dtype=np.byte)

    def generate_neigh_indexes_squashed(self):
        neigh_range = self.neigh_range
        size = 3 + (neigh_range - 1) * 2
        neigh_shape = (size, size, 3)
        temp = np.ones(neigh_shape, dtype=int)

        flat_ind = np.array(self.ind_decompose_flat_z)
        flat_ind = flat_ind.transpose()
        flat_ind[0] += neigh_range
        flat_ind[1] += neigh_range
        flat_ind[2] += 1

        temp[flat_ind[0], flat_ind[1], flat_ind[2]] = 0

        coord = np.array(np.nonzero(temp))
        coord[0] -= neigh_range
        coord[1] -= neigh_range
        coord[2] -= 1
        coord = coord.transpose()

        coord = np.concatenate((self.ind_decompose_flat_z, coord))

        return np.array(coord, dtype=np.byte)

    def generate_neigh_indexes_flat(self):
        neigh_range = self.neigh_range
        size = 3 + (neigh_range - 1) * 2
        neigh_shape = (size, size, 3)
        temp = np.ones(neigh_shape, dtype=int)
        temp[:, :, 0] = 0
        temp[:, :, 2] = 0

        flat_ind = np.array(self.ind_decompose_flat_z)
        flat_ind = flat_ind.transpose()
        flat_ind[0] += neigh_range
        flat_ind[1] += neigh_range
        flat_ind[2] += 1

        temp[flat_ind[0], flat_ind[1], flat_ind[2]] = 0

        coord = np.array(np.nonzero(temp))
        coord[0] -= neigh_range
        coord[1] -= neigh_range
        coord[2] -= 1
        coord = coord.transpose()

        coord = np.concatenate((self.ind_decompose_flat_z, coord))
        additional = coord[[2, 5]]
        coord = np.delete(coord, [2, 5], axis=0)
        coord = np.concatenate((coord, additional))

        return np.array(coord, dtype=np.byte)

    def print_static_params_to_file(self, cls, file_obj, indent=0):
        snapshot = self._snapshot_static_params(cls)
        file_obj.write(self._format_pretty_snapshot(snapshot))

    def print_static_params(self, cls, indent=0):
        snapshot = self._snapshot_static_params(cls)
        print(self._format_pretty_snapshot(snapshot))

    def _snapshot_static_params(self, obj, _visited=None):
        if _visited is None:
            _visited = set()

        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {
                str(key): self._snapshot_static_params(value, _visited)
                for key, value in obj.items()
            }
        if isinstance(obj, (list, tuple, set)):
            return [self._snapshot_static_params(item, _visited) for item in obj]

        obj_id = id(obj)
        if obj_id in _visited:
            return "<recursion>"

        if hasattr(obj, "__dict__"):
            _visited.add(obj_id)
            mapped = {}
            for attr_name, attr_value in vars(obj).items():
                if callable(attr_value) or attr_name.startswith('__'):
                    continue
                mapped[attr_name] = self._snapshot_static_params(attr_value, _visited)
            _visited.remove(obj_id)
            return mapped

        if isinstance(obj, type):
            _visited.add(obj_id)
            mapped = {}
            for attr_name, attr_value in obj.__dict__.items():
                if callable(attr_value) or attr_name.startswith('__'):
                    continue
                mapped[attr_name] = self._snapshot_static_params(attr_value, _visited)
            _visited.remove(obj_id)
            return mapped

        return repr(obj)

    @staticmethod
    def _format_pretty_snapshot(snapshot):
        return pprint.pformat(snapshot, indent=2, width=120, sort_dicts=False)
