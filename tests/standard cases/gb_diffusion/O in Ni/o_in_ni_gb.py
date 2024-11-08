from engine import *
import inspect

if __name__ == '__main__':

    class NewSystem(SimulationConfigurator):
        def __init__(self):
            super().__init__()
            self.cases.first.microstructure = voronoi.VoronoiMicrostructure(Config.N_CELLS_PER_AXIS)
            self.cases.first.microstructure.generate_voronoi_3d(50)
            self.cases.first.microstructure.show_microstructure()
            self.save_microstructure(self.cases.first.microstructure)
            self.cases.first.oxidant.microstructure = self.cases.first.microstructure
            self.cases.first.oxidant.diffuse = self.cases.first.oxidant.diffuse_gb

    source_code = inspect.getsource(NewSystem)
    Config.INITIAL_SCRIPT += source_code
    Config.COMMENT = """This script simulates inward diffusion of O in Ni with GB diffusion along the interfaces created
    via Voronoi tesselation"""
    new_system = NewSystem()

    try:
        new_system.start_simulation()
    finally:
        new_system.save_results()
        new_system.insert_last_it()
        new_system.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", new_system.c_automata.iteration)
        print("____________________________________________________________")
        print()
