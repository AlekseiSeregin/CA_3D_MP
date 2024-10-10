from engine import *


if __name__ == '__main__':

    Config.COMMENT = """ """

    new_system = SimulationConfigurator()
    new_system.configurate_functions_lut_nicr5()

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
