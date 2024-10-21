from elements import dummy
from engine import *
import inspect

if __name__ == "__main__":

    class NewSystem(SimulationConfigurator):
        def __init__(self):
            super().__init__()
            self.cases.first.oxidant.diffuse = self.cases.first.oxidant.diffuse_bulk

            self.c_automata.cur_case = self.cases.first
            self.c_automata.cur_case_mp = self.cases.first_mp

            # self.save_function = self.save_results_only_inw
            self.save_function = False


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

        new_system.insert_last_it()
        new_system.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", new_system.c_automata.iteration)
        print("____________________________________________________________")
        print()
