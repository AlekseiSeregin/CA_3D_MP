from new_engine import *


if __name__ == '__main__':
    Config.COMMENT = """

    Nucleation and dissolution throughout the whole simulation (both schemes applied). Also with kinetic coefficient!!!
    Go along the kinetic growth line! Check the kinetic file as well!!!
"""

    new_system = SimulationConfigurator()
    new_system.configurate_functions()

    try:
        new_system.run_simulation()
    finally:
        try:
            if not Config.SAVE_WHOLE:
                new_system.save_results()

        except (Exception,):
            # new_system.save_results()
            print("Not SAVED!")

        # new_system.save_results()
        new_system.terminate_workers()
        new_system.unlink()

        cumul_prod = new_system.ca.cumul_prod.get_buffer()
        growth_rate = new_system.ca.growth_rate.get_buffer()

        # Transpose the arrays to switch rows and columns
        cumul_prod_transposed = cumul_prod.T
        growth_rate_transposed = growth_rate.T

        # Interleave the columns
        interleaved_array = np.empty((new_system.ca.cumul_prod.last_in_buffer, 2 * new_system.ca.cells_per_axis), dtype=float)
        interleaved_array[:, 0::2] = cumul_prod_transposed
        interleaved_array[:, 1::2] = growth_rate_transposed

        iterations = np.arange(new_system.ca.cumul_prod.last_in_buffer) * Config.STRIDE

        data = np.column_stack((iterations.T, interleaved_array))

        output_file_path = Config.SAVE_PATH + Config.GENERATED_VALUES.DB_ID + "_kinetics.txt"
        with open(output_file_path, "w", encoding='utf-8') as f:
            for row in data:
                f.write(" ".join(map(str, row)) + "\n")

        # data = np.column_stack(
        #     (iterations, new_system.ca.cumul_prod.get_buffer(), new_system.ca.growth_rate.get_buffer()))
        # output_file_path = Config.SAVE_PATH + Config.GENERATED_VALUES.DB_ID + "_kinetics.txt"
        # with open(output_file_path, "w") as f:
        #     for row in data:
        #         f.write(" ".join(map(str, row)) + "\n")

        #       data = np.column_stack(
        #     (iterations, new_system.ca.cumul_prod1.get_buffer(), new_system.ca.growth_rate1.get_buffer()))
        # output_file_path = Config.SAVE_PATH + Config.GENERATED_VALUES.DB_ID + "1" + "_kinetics2.txt"
        # with open(output_file_path, "w") as f:
        #     for row in data:
        #         f.write(" ".join(map(str, row)) + "\n")

        new_system.insert_last_it()
        new_system.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", new_system.ca.iteration)
        print("____________________________________________________________")
        print()
        # traceback.print_exc()


