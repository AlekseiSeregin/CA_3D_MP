from elements import dummy
from engine import *
import inspect

if __name__ == '__main__':
    def some_scope(new_p0, new_p1):

        Config.PROBABILITIES.PRIMARY.p0 = new_p0
        Config.PROBABILITIES.PRIMARY.p1 = new_p1

        class NewSystem(SimulationConfigurator):
            def __init__(self):
                super().__init__()
                self.c_automata.cases.first.oxidant.diffuse = self.c_automata.cases.first.oxidant.diffuse_bulk
                self.c_automata.cases.first.active.diffuse = elements.diffuse_bulk_mp

                self.c_automata.get_cur_ioz_bound = self.c_automata.ioz_depth_furthest_inward

                self.c_automata.precip_func = self.c_automata.precipitation_current_case
                self.c_automata.get_combi_ind = self.c_automata.get_combi_ind_standard

                self.c_automata.cases.first_mp.precip_step = precip_step_standard
                self.c_automata.cases.first_mp.check_intersection = ci_single

                self.save_function = self.calc_precipitation_front_only_cells

                self.c_automata.cases.first_mp.nucleation_probabilities = utils.NucleationProbabilities(
                    Config.PROBABILITIES.PRIMARY,
                    Config.PRODUCTS.PRIMARY)

        source_code = inspect.getsource(NewSystem)
        Config.INITIAL_SCRIPT = "\n" + source_code
        new_system = NewSystem()

        try:
            new_system.run_simulation()
        finally:
            new_system.save_results()
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
            print("____________________________________________________________")
            print(new_p0, " ", new_p1,  " done!")
            print("____________________________________________________________")

    # p_list = [[0.01, 1],
    #           [0.001, 1],
    #           [0.0001, 1],
    #           [0.00001, 1],
    #           [0.000001, 1],
    #           [0.0000001, 1],
    #
    #           [0.01, 0.1],
    #
    #           [0.001, 0.1],
    #           [0.001, 0.01],
    #
    #           [0.0001, 0.1],
    #           [0.0001, 0.01],
    #           [0.0001, 0.001],
    #
    #           [0.00001, 0.1],
    #           [0.00001, 0.01],
    #           [0.00001, 0.001],
    #           [0.00001, 0.0001],
    #
    #           [0.000001, 0.1],
    #           [0.000001, 0.01],
    #           [0.000001, 0.001],
    #           [0.000001, 0.0001],
    #           [0.000001, 0.00001],
    #
    #           [0.0000001, 0.1],
    #           [0.0000001, 0.01],
    #           [0.0000001, 0.001],
    #           [0.0000001, 0.0001],
    #           [0.0000001, 0.00001],
    #           [0.0000001, 0.000001]]

    p_list = [[0.0000001, 0.00001],
              [0.0000001, 0.000001]]

    for item in p_list:
        some_scope(item[0], item[1])
