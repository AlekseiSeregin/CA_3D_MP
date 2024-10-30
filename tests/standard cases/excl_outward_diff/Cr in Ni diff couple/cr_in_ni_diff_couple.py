from engine import *
import inspect

if __name__ == '__main__':

    class NewSystem(SimulationConfigurator):
        def __init__(self):
            super().__init__()
            self.c_automata.cases.first.active.diffuse = elements.diffuse_bulk_mp

    source_code = inspect.getsource(NewSystem)
    Config.INITIAL_SCRIPT += source_code
    Config.COMMENT = "This script simulates outward diffusion of Cr in Ni as diffusion couple of Ni + Ni-20at%Cr"
    new_system = NewSystem()

    try:
        new_system.run_simulation()
    finally:

        new_system.save_results()
        new_system.terminate_workers()
        new_system.unlink()
        new_system.insert_last_it()
        new_system.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", new_system.c_automata.iteration)
        print("____________________________________________________________")
        print()