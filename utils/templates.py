import numpy as np


class CaseSetUp:
    def __init__(self):
        self.oxidant = None
        self.active = None
        self.product = None
        self.prod_indexes = None
        self.product_ind_not_stab = None
        self.fix_init_precip_func_ref = None
        self.precip_3d_init = None
        self.shm_pool = {"product_indexes": None,
                         "product_ind_not_stab": None,
                         "precip_3d_init": None}

    def close_and_unlink_shared_memory(self):
        for key, shm in self.shm_pool.items():
            if shm is not None:
                shm.close()
                shm.unlink()


class CaseSetUpMP:
    def __init__(self):
        self.active_c3d_shm_mdata = None
        self.active_cells_shm_mdata = None
        self.active_dirs_shm_mdata = None

        self.oxidant_c3d_shm_mdata = None

        self.product_c3d_shm_mdata = None
        self.oxidation_number = None
        self.full_shm_mdata = None
        self.to_check_with_shm_mdata = None
        self.prod_indexes_shm_mdata = None
        self.prod_indexes_not_stab_shm_mdata = None

        self.go_around_func_ref = None
        self.precip_3d_init_shm_mdata = None

        self.nucleation_probabilities = None
        self.dissolution_probabilities = None

        self.precip_step = None
        self.check_intersection = None

        self.decomposition = None
        self.fix_full_cells = None

        self.threshold_inward = None
        self.threshold_outward = None

        self.cells_per_axis = None


class CaseRef:
    def __init__(self):
        self.first = CaseSetUp()
        self.first_mp = CaseSetUpMP()
        self.second = CaseSetUp()
        self.second_mp = CaseSetUpMP()
        self.third = CaseSetUp()
        self.third_mp = CaseSetUpMP()
        self.fourth = CaseSetUp()
        self.fourth_mp = CaseSetUpMP()
        self.fifth = CaseSetUp()
        self.fifth_mp = CaseSetUpMP()

        self.accumulated_products = None
        self.accumulated_products_shm = None
        self.accumulated_products_shm_mdata = None

    def close_shms(self):
        self.first.close_and_unlink_shared_memory()
        self.second.close_and_unlink_shared_memory()
        self.third.close_and_unlink_shared_memory()
        self.fourth.close_and_unlink_shared_memory()
        self.fifth.close_and_unlink_shared_memory()

        if self.accumulated_products_shm is not None:
            self.accumulated_products_shm.close()
            self.accumulated_products_shm.unlink()

    def reaccumulate_products(self, exclude_case):
        np.add(self.first.product.c3d, self.second.product.c3d, out=self.accumulated_products, dtype=np.ubyte)
        np.add(self.accumulated_products, self.third.product.c3d, out=self.accumulated_products, dtype=np.ubyte)
        np.add(self.accumulated_products, self.fourth.product.c3d, out=self.accumulated_products, dtype=np.ubyte)
        np.add(self.accumulated_products, self.fifth.product.c3d, out=self.accumulated_products, dtype=np.ubyte)
        np.subtract(self.accumulated_products, exclude_case.product.c3d, out=self.accumulated_products, dtype=np.ubyte)

    def reaccumulate_products_no_exclusion(self):
        np.add(self.first.product.c3d, self.second.product.c3d, out=self.accumulated_products, dtype=np.ubyte)
        np.add(self.accumulated_products, self.third.product.c3d, out=self.accumulated_products, dtype=np.ubyte)
        np.add(self.accumulated_products, self.fourth.product.c3d, out=self.accumulated_products, dtype=np.ubyte)
        np.add(self.accumulated_products, self.fifth.product.c3d, out=self.accumulated_products, dtype=np.ubyte)


DEFAULT_PARAM = {
    "oxidant": {"primary": {"elem": "N",
                            "diffusion_condition": "N in Ni20Cr2Ti Krupp",
                            "cells_concentration": 0.01},
                "secondary": {"elem": "None",
                              "diffusion_condition": "Test",
                              "cells_concentration": 0.1}},

    "active_element": {"primary": {"elem": "Ti",
                                   "diffusion_condition": "Ti in Ni Krupp",
                                   "mass_concentration": 0.02,
                                   "cells_concentration": 0.04},
                       "secondary": {"elem": "None",
                                     "diffusion_condition": "Test",
                                     "mass_concentration": 0.02,
                                     "cells_concentration": 0.02}
                       },
    "matrix_elem": {"elem": "Ni",
                    "diffusion_condition": "not_used",
                    "concentration": 0},

    "full_cells": False,
    "diff_in_precipitation": 3.05 * 10 ** -14,
    "diff_out_precipitation": 3.05 * 10 ** -14,
    "temperature": 1000,
    "n_cells_per_axis": 102,
    "n_iterations": 1000,
    "stride": 1,
    "sim_time": 720000,
    "size": 0.0005,

    "threshold_inward": 1,
    "threshold_outward": 1,
    "sol_prod": 5.621 * 10 ** -10,

    "nucleation_probability": 1,
    "het_factor": 300,

    "dissolution_p": 0.1,
    "dissolution_n": 2,
    "exponent_power": 3,
    "block_scale_factor": 2,

    "inward_diffusion": True,
    "outward_diffusion": True,
    "compute_precipitations": True,
    "diffusion_in_precipitation": None,

    "save_whole": False,
    "save_path": 'W:/SIMCA/test_runs_data/',

    "neigh_range": 1,
    "decompose_precip": False,

    "phase_fraction_lim": 0.123456789,
    "hf_deg_lim": 0.123456789,

    "lowest_neigh_numb": 0.123456789,
    "final_nucl_prob": 0.123456789,

    "min_dissol_prob": 0.123456789,
    "het_factor_dissolution": 0.123456789,

    "final_dissol_prob": 0.123456789,
    "final_het_factor_dissol": 0.123456789,

    "final_min_dissol_prob": 0.123456789,

    "max_neigh_numb": 0.123456789,

    "product_kinetic_const": 0.123456789,

    "error_prod_conc": 0.123456789,

    "init_P1": 0.123456789,
    "final_P1": 0.123456789,
    "b_const_P1": 0.123456789,

    "nucl_adapt_function": 0.123456789,
    "dissol_adapt_function": 0.123456789,

    "init_P1_diss": 0.123456789,
    "final_P1_diss": 0.123456789,
    "b_const_P1_diss": 0.123456789,

    "b_const_P0_nucl": 0.123456789,

    "bend_b_init":  0.123456789,
    "bend_b_final":  0.123456789,
}