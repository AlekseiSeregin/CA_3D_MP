from engine import *
import inspect

if __name__ == '__main__':

    class NewSystem(SimulationConfigurator):
        def __init__(self):
            super().__init__()

            self.cases.first.oxidant.diffuse = self.cases.first.oxidant.diffuse_bulk
            self.cases.first.active.diffuse = elements.diffuse_bulk_mp

            self.c_automata.get_cur_ioz_bound = self.c_automata.ioz_depth_from_kinetics
            # self.c_automata.get_cur_dissol_ioz_bound = self.c_automata.ioz_dissolution_where_prod

            self.c_automata.precip_func = self.c_automata.precipitation_current_case
            self.c_automata.get_combi_ind = self.c_automata.get_combi_ind_standard

            self.c_automata.cases.first_mp.precip_step = precip_step_standard
            self.c_automata.cases.first_mp.check_intersection = ci_single

            # self.cases.first.fix_init_precip_func_ref = self.c_automata.fix_init_precip_dummy

            self.c_automata.decomposition = self.c_automata.dissolution_standard
            # self.c_automata.decomposition = None
            self.c_automata.decomposition_intrinsic = self.c_automata.simple_decompose_mp
            self.c_automata.cases.first_mp.decomposition = dissolution_zhou_wei_no_bsf

            self.c_automata.cur_case = self.cases.first
            self.c_automata.cur_case_mp = self.cases.first_mp

            self.c_automata.cases.first_mp.nucleation_probabilities = utils.NucleationProbabilities(
                Config.PROBABILITIES.PRIMARY,
                Config.PRODUCTS.PRIMARY)
            self.c_automata.cases.first_mp.dissolution_probabilities = utils.DissolutionProbabilities(
                Config.PROBABILITIES.PRIMARY)

            # self.save_function = self.save_results_only_prod_prime
            self.save_function = None


    source_code = inspect.getsource(NewSystem)
    Config.INITIAL_SCRIPT += source_code

    new_system = NewSystem()

    try:
        new_system.run_simulation()
    finally:
        # try:
        #     print("Try!")
        #     # if not Config.SAVE_WHOLE:
        #
        # except (Exception,):
        #     new_system.save_results()
        #     print("Not SAVED Exeption!")

        new_system.save_results()
        print("SAVED!")

        new_system.terminate_workers()
        new_system.unlink()

        cumul_prod = new_system.c_automata.cumul_prod.get_buffer()
        growth_rate = new_system.c_automata.growth_rate.get_buffer()

        # Transpose the arrays to switch rows and columns
        cumul_prod_transposed = cumul_prod.T
        growth_rate_transposed = growth_rate.T

        # Interleave the columns
        interleaved_array = np.empty((new_system.c_automata.cumul_prod.last_in_buffer, 2 * new_system.c_automata.cells_per_axis),
                                     dtype=float)
        interleaved_array[:, 0::2] = cumul_prod_transposed
        interleaved_array[:, 1::2] = growth_rate_transposed

        iterations = np.arange(new_system.c_automata.cumul_prod.last_in_buffer) * Config.STRIDE

        data = np.column_stack((iterations.T, interleaved_array))

        output_file_path = Config.SAVE_PATH + Config.GENERATED_VALUES.DB_ID + "_kinetics.txt"
        with open(output_file_path, "w", encoding='utf-8') as f:
            for row in data:
                f.write(" ".join(map(str, row)) + "\n")

        new_system.insert_last_it()
        new_system.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", new_system.c_automata.iteration)
        print("____________________________________________________________")
        print()
