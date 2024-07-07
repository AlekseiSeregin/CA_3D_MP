import copy
from old_scripts.old_engine import *
import traceback

if __name__ == '__main__':

    user_input = {"oxidant": {"primary": {"elem": "O",
                                          "diffusion_condition": "O in Ni Krupp",
                                          "cells_concentration": 0.0012},
                              "secondary": {"elem": "None",
                                            "diffusion_condition": "N in Ni Krupp",
                                            "cells_concentration": 0.1}
                              },

                  "active_element": {"primary": {"elem": "Al",
                                                 "diffusion_condition": "Al in Ni Krupp",
                                                 "mass_concentration": 0.025,
                                                 "cells_concentration": 0.2},
                                     "secondary": {"elem": "None",
                                                   "diffusion_condition": "Al in Ni Krupp",
                                                   "mass_concentration": 0.025,
                                                   "cells_concentration": 0.2}
                                     },

                  "matrix_elem": {"elem": "Ni",
                                  "diffusion_condition": "not_used",
                                  "concentration": 0},

                  "full_cells": False,
                  "diff_in_precipitation": 3.05 * 10 ** -14,  # [m2/sek]
                  "diff_out_precipitation": 3.05 * 10 ** -14,  # [m2/sek]
                  "temperature": 1100,  # °C
                  "n_cells_per_axis": 102,  # ONLY MULTIPLES OF 3+(neigh_range-1)*2 ARE ALLOWED
                  "n_iterations": 5000000,  # must be >= n_cells_per_axis
                  "stride": 500,  # n_iterations / stride = n_iterations for outward diffusion
                  "sim_time": 1080000,  # [sek]
                  "size": 10000 * (10**-6),  # [m]

                  "threshold_inward": 1,
                  "threshold_outward": 1,
                  "sol_prod": 6.25 * 10 ** -31,  # 5.621 * 10 ** -10

                  "nucleation_probability": 0.1,
                  "het_factor": 10**0.5,  # not used anymore

                  "dissolution_p": 1 * 10**-3,
                  "dissolution_n": 2,  # not used anymore
                  "exponent_power": 0,  # not used anymore
                  "block_scale_factor": 1,

                  "inward_diffusion": True,
                  "outward_diffusion": True,
                  "compute_precipitations": True,
                  "diffusion_in_precipitation": False,

                  "save_whole": False,
                  "save_path": 'C:/test_runs_data/',

                  "neigh_range": 1,  # neighbouring ranges    1, 2, 3, 4, 5,  6,  7,  8,  9,  10
                                     #          and           |  |  |  |  |   |   |   |   |   |
                                     # corresponding divisors 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
                  "decompose_precip": True,

                  "phase_fraction_lim": 0.07,
                  "hf_deg_lim": 10**10,  # not used anymore
                  "lowest_neigh_numb": 16,  # not used anymore
                  "final_nucl_prob": 1*10**-1,

                  "min_dissol_prob": 1 * 10 ** -5,
                  "het_factor_dissolution": 10 ** 1,  # not used anymore
                  "final_dissol_prob": 1 * 10 ** 0,
                  "final_het_factor_dissol": 10 ** 0,  # not used anymore
                  "final_min_dissol_prob": 1 * 10 ** -4,

                  "max_neigh_numb": 0,
                  "product_kinetic_const": 0.0000003,  # not used anymore
                  "error_prod_conc": 1.01,  # not used anymore

                  "init_P1": 5 * 10 ** -1,
                  "final_P1": 1 * 10 ** -3,
                  "b_const_P1": -3,

                  "nucl_adapt_function": 3,
                  "dissol_adapt_function": 5,

                  "init_P1_diss": 5 * 10 ** -4,
                  "final_P1_diss": 1 * 10 ** 0,
                  "b_const_P1_diss": 600,

                  "b_const_P0_nucl": 1,

                  "bend_b_init": 0.03,
                  "bend_b_final": -20,

                  }

    backup_user_input = copy.deepcopy(user_input)
    eng = CellularAutomata(user_input=user_input)

    eng.precip_func = eng.precipitation_first_case
    eng.get_combi_ind = eng.get_combi_ind_atomic_opt_for_growth

    eng.decomposition = eng.dissolution_test

    eng.cur_case = eng.cases.first

    try:
        eng.simulation()
    finally:
        try:
            if not user_input["save_whole"]:
                eng.save_results()

        except (Exception,):
            backup_user_input["save_path"] = "C:/test_runs_data/"
            eng.utils = Utils(backup_user_input)
            eng.utils.create_database()
            eng.utils.generate_param()
            eng.save_results()
            print()
            print("____________________________________________________________")
            print("Saving To Standard Folder Crashed!!!")
            print("Saved To ->> C:/test_runs_data/!!!")
            print("____________________________________________________________")
            print()

        eng.insert_last_it()
        eng.utils.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", eng.iteration)
        print("____________________________________________________________")
        print()
        traceback.print_exc()
