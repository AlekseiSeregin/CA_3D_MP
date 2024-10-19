from elements import dummy
from engine import *
import inspect

if __name__ == '__main__':

    class NewSystem(SimulationConfigurator):
        def __init__(self):
            super().__init__()
            self.c_automata.cases.first.oxidant.diffuse = dummy
            self.c_automata.cases.first.active.diffuse = dummy
            
            self.c_automata.get_cur_ioz_bound = self.c_automata.ioz_depth_furthest_inward

            self.c_automata.precip_func = self.c_automata.precipitation_current_case
            self.c_automata.get_combi_ind = self.c_automata.get_combi_ind_standard

            self.c_automata.cases.first_mp.precip_step = precip_step_standard
            self.c_automata.cases.first_mp.check_intersection = ci_single_no_growth

            self.cases.first.fix_init_precip_func_ref = self.c_automata.fix_init_precip_dummy

            self.c_automata.cur_case = self.cases.first
            self.c_automata.cur_case_mp = self.cases.first_mp

            self.save_function = None

    source_code = inspect.getsource(NewSystem)
    Config.INITIAL_SCRIPT += source_code

    new_system = NewSystem()

    try:
        new_system.run_simulation()
    finally:

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
