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
    new_system.start_simulation()
