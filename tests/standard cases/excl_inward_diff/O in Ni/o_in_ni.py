from engine import *
import inspect

if __name__ == '__main__':
    Config.COMMENT = "This script simulates inward diffusion of O in Ni"

    class NewSystem(SimulationConfigurator):
        def __init__(self):
            super().__init__()
            self.c_automata.cases.first.oxidant.diffuse = self.c_automata.cases.first.oxidant.diffuse_bulk

    source_code = inspect.getsource(NewSystem)
    Config.INITIAL_SCRIPT += source_code
    new_system = NewSystem()
    new_system.start_simulation()
