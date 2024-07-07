from engine import *
import traceback
from configuration import Config

if __name__ == '__main__':

    Config.COMMENT = """

    eng.primary_oxidant.diffuse = eng.primary_oxidant.diffuse_with_scale
    eng.primary_active.diffuse = eng.primary_active.diffuse_with_scale
    eng.precip_func = eng.precipitation_first_case
    eng.get_combi_ind = eng.get_combi_ind_standard
    eng.precip_step = eng.precip_step_standard
    eng.check_intersection = eng.ci_single_no_growth
    eng.decomposition = eng.dissolution_atomic_stop_if_no_active_or_no_oxidant
    (inside this function the dissolution_zhou_wei_with_bsf_aip() was called!!!!)

    eng.cur_case = eng.cases.first
    eng.cases.first.go_around_func_ref = eng.go_around_mult_oxid_n_also_partial_neigh_aip

    eng.cur_case.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                            Config.PRODUCTS.PRIMARY)
                             
    Script name: super_serie.py
    

"""

    eng = CellularAutomata()

    eng.primary_oxidant.diffuse = eng.primary_oxidant.diffuse_bulk
    eng.primary_active.diffuse = eng.primary_active.diffuse_bulk
    eng.precip_func = eng.precipitation_first_case
    eng.get_combi_ind = eng.get_combi_ind_standard
    eng.precip_step = eng.precip_step_standard
    eng.check_intersection = eng.ci_single_no_growth_only_p0

    eng.cur_case = eng.cases.first
    eng.cases.first.fix_init_precip_func_ref = eng.fix_init_precip_dummy


    try:
        eng.simulation()
    finally:
        try:
            if not Config.SAVE_WHOLE:
                eng.save_results()

        except (Exception,):
            eng.save_results()
            print("Not SAVED!")
        #     backup_user_input["save_path"] = "C:/test_runs_data/"
        #     eng.utils = Utils(backup_user_input)
        #     eng.utils.create_database()
        #     eng.utils.generate_param()
        #     eng.save_results()
        #     print()
        #     print("____________________________________________________________")
        #     print("Saving To Standard Folder Crashed!!!")
        #     print("Saved To ->> C:/test_runs_data/!!!")
        #     print("____________________________________________________________")
        #     print()
        #
        #     # data = np.column_stack(
        #     #     (np.arange(eng.iteration), eng.cumul_prod[:eng.iteration]))
        #     # output_file_path = "W:/SIMCA/test_runs_data/" + eng.utils.param["db_id"] + ".txt"
        #     # with open(output_file_path, "w") as f:
        #     #     for row in data:
        #     #         f.write(" ".join(map(str, row)) + "\n")
        eng.insert_last_it()
        eng.utils.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", eng.iteration)
        print("____________________________________________________________")
        print()
        traceback.print_exc()
