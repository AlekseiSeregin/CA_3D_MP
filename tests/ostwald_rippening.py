from engine import *
import traceback
from configuration import Config, save_script_contents_as_string


if __name__ == '__main__':

    save_script_contents_as_string(__file__, Config)
    # user_input = {"oxidant": {"primary": {"elem": "O",
    #                                       "diffusion_condition": "O in Ni Krupp",
    #                                       "cells_concentration": 0.1},
    #                           "secondary": {"elem": "None",
    #                                         "diffusion_condition": "N in Ni Krupp",
    #                                         "cells_concentration": 0.3}
    #                           },
    #
    #               "active_element": {"primary": {"elem": "Al",
    #                                              "diffusion_condition": "O in Ni Krupp",
    #                                              "mass_concentration": 0.03,
    #                                              "cells_concentration": 0.1},
    #                                  "secondary": {"elem": "None",
    #                                                "diffusion_condition": "Al in Ni Krupp",
    #                                                "mass_concentration": 0.0,
    #                                                "cells_concentration": 0.0}
    #                                  },
    #
    #               "matrix_elem": {"elem": "Ni",
    #                               "diffusion_condition": "not_used",
    #                               "concentration": 0},
    #
    #               "full_cells": False,
    #               "diff_in_precipitation": 3.05 * 10 ** -14,  # [m2/sek]
    #               "diff_out_precipitation": 3.05 * 10 ** -14,  # [m2/sek]
    #               "temperature": 1100,  # °C
    #               "n_cells_per_axis": 102,  # ONLY MULTIPLES OF 3+(neigh_range-1)*2 ARE ALLOWED
    #               "n_iterations": 300000,  # must be >= n_cells_per_axis
    #               "stride": 1,  # n_iterations / stride = n_iterations for outward diffusion
    #               "sim_time": 72000,  # [sek]
    #               "size": 70 * (10**-6),  # [m]
    #
    #               "threshold_inward": 1,
    #               "threshold_outward": 1,
    #               "sol_prod": 6.25 * 10 ** -31,  # 5.621 * 10 ** -10
    #
    #               "nucleation_probability": 0,
    #               "het_factor": 10**0.5,  # not used anymore
    #
    #               "dissolution_p": 1 * 10 ** 0,
    #               "dissolution_n": 2,  # not used anymore
    #               "exponent_power": 0,  # not used anymore
    #               "block_scale_factor": 1,
    #
    #               "inward_diffusion": True,
    #               "outward_diffusion": True,
    #               "compute_precipitations": True,
    #               "diffusion_in_precipitation": False,
    #
    #               "save_whole": False,
    #               "save_path": 'C:/test_runs_data/',
    #
    #               "neigh_range": 1,  # neighbouring ranges    1, 2, 3, 4, 5,  6,  7,  8,  9,  10
    #                                  #          and           |  |  |  |  |   |   |   |   |   |
    #                                  # corresponding divisors 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
    #               "decompose_precip": True,
    #
    #               "phase_fraction_lim": 0.6,
    #               "hf_deg_lim": 10**10,
    #               "lowest_neigh_numb": 16,
    #               "final_nucl_prob": 1*10**0,
    #
    #               "min_dissol_prob": 1 * 10 ** -2.4,
    #               "het_factor_dissolution": 10 ** 1,  # not used anymore
    #               "final_dissol_prob": 1 * 10 ** 0,
    #               "final_het_factor_dissol": 10 ** 0,  # not used anymore
    #               "final_min_dissol_prob": 1 * 10 ** -1,
    #
    #               "max_neigh_numb": 0,
    #               "product_kinetic_const": 0.0000003,  # not used anymore
    #               "error_prod_conc": 1.01,  # not used anymore
    #
    #               "init_P1": 0.9999,
    #               "final_P1": 1 * 10 ** 0,
    #               "b_const_P1": 1,
    #
    #               "nucl_adapt_function": 3,
    #               "dissol_adapt_function": 5,
    #
    #               "init_P1_diss": 0.9999,
    #               "final_P1_diss": 1 * 10 ** 0,
    #               "b_const_P1_diss": 1,
    #
    #               "b_const_P0_nucl": 1,
    #
    #               "bend_b_init": 0.5,
    #               "bend_b_final": 0.1,
    #
    #               }

    Config.COMMENT = """

        Nucleation and dissolution throughout the whole simulation (both schemes applied). Also with kinetic coeeficient!!!
        Go along the kinetic growth line! Check the kinetic file as well!!!

        CHANGED THE SCHEMES OF NUCLEATION AND DISSOLUTION ->>> NOW ALSO THE PARTIAL NEIGHBOURS ARE CONSIDERED!!!!

    """

    eng = CellularAutomata()

    eng.primary_oxidant.diffuse = eng.primary_oxidant.diffuse_with_scale
    eng.primary_active.diffuse = eng.primary_active.diffuse_with_scale

    eng.precip_func = eng.precipitation_first_case
    eng.get_combi_ind = eng.get_combi_ind_standard
    eng.precip_step = eng.precip_step_standard
    eng.check_intersection = eng.ci_single

    eng.decomposition = eng.dissolution_test
    eng.decomposition_intrinsic = eng.dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL

    eng.cur_case = eng.cases.first
    eng.cases.first.go_around_func_ref = eng.go_around_mult_oxid_n_also_partial_neigh_aip

    eng.cur_case.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                          Config.PRODUCTS.PRIMARY)
    eng.cur_case.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                            Config.PRODUCTS.PRIMARY)

    n_per_page = eng.primary_oxidant.n_per_page
    eng.primary_oxidant.cells = np.random.randint(Config.N_CELLS_PER_AXIS, size=(3, int(n_per_page * Config.N_CELLS_PER_AXIS)), dtype=np.short)
    eng.primary_oxidant.dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(eng.primary_oxidant.cells[0]))
    eng.primary_oxidant.dirs = np.array(np.unravel_index(eng.primary_oxidant.dirs, (3, 3, 3)), dtype=np.byte)
    eng.primary_oxidant.dirs -= 1

    eng.cases.first.oxidant = eng.primary_oxidant

    eng.primary_oxidant.transform_to_3d()

    eng.primary_active.transform_to_3d(Config.N_CELLS_PER_AXIS)
    eng.cases.first.active = eng.primary_active

    # set product
    # first sphere
    half_thickness = 7
    middle = int(Config.N_CELLS_PER_AXIS / 2)
    minus = middle - half_thickness
    plus = middle + half_thickness
    shift = 0

    for x in range(minus, plus):
        for y in range(minus, plus):
            for z in range(minus, plus):
                if (x - middle) ** 2 + (y - middle) ** 2 + (z - middle) ** 2 <= half_thickness ** 2:
                    eng.primary_product.c3d[z, y, x + shift] = eng.primary_oxid_numb
                    eng.primary_product.full_c3d[z, y, x + shift] = True
                    eng.product_x_nzs[x + shift] = True
                    eng.primary_oxidant.c3d[z, y, x + shift] = 0
                    eng.primary_active.c3d[z, y, x + shift] = 0

    # # second sphere
    # half_thickness = 5
    # shift = -20
    #
    # for x in range(minus, plus):
    #     for y in range(minus, plus):
    #         for z in range(minus, plus):
    #             if (x - middle) ** 2 + (y - middle) ** 2 + (z - middle) ** 2 <= half_thickness ** 2:
    #                 eng.primary_product.c3d[z, y, x + shift] = eng.primary_oxid_numb
    #                 eng.primary_product.full_c3d[z, y, x + shift] = True
    #                 eng.product_x_nzs[x + shift] = True
    #                 eng.primary_oxidant.c3d[z, y, x + shift] = 0
    #                 eng.primary_active.c3d[z, y, x + shift] = 0

    eng.primary_oxidant.transform_to_descards()
    eng.primary_active.transform_to_descards()




    try:
        eng.simulation()
    finally:
        try:
            if not Config.SAVE_WHOLE:
                eng.save_results()

        except (Exception,):
            eng.save_results()
            print("Not SAVED!")
        iterations = np.arange(eng.cumul_prod.last_in_buffer) * eng.precipitation_stride

        data = np.column_stack(
            (iterations, eng.cumul_prod.get_buffer(), eng.growth_rate.get_buffer()))
        output_file_path = "C:/test_runs_data/" + Config.GENERATED_VALUES.DB_ID + "_kinetics.txt"
        with open(output_file_path, "w") as f:
            for row in data:
                f.write(" ".join(map(str, row)) + "\n")

        eng.insert_last_it()
        eng.utils.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", eng.iteration)
        print("____________________________________________________________")
        print()
        traceback.print_exc()

