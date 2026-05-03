from engine import *
import inspect

if __name__ == '__main__':

    class NewSystem(SimulationConfigurator):
        def __init__(self):
            super().__init__()

            self.c_automata.precip_func = self.c_automata.nucleate
            
            self.c_automata.get_cur_ioz_bound = self.c_automata.ioz_depth_furthest_inward

            if bool(getattr(Config, "USE_JMATPRO_BLOCKS_IGNITED", False)):
                self.c_automata.get_combi_ind = self.c_automata.get_comb_ind_jmatpro_blocks_ignited
            else:
                self.c_automata.get_combi_ind = self.c_automata.get_comb_ind_jmatpro_generic

            self.save_function = self.save_results_inward_only
        
            # self.c_automata.decomposition = self.c_automata.dissolve

    source_code = inspect.getsource(NewSystem)
    Config.INITIAL_SCRIPT += source_code
    Config.COMMENT = "This script simulates outward diffusion of Cr in Ni as diffusion couple of Ni + Ni-20at%Cr"
    new_system = NewSystem()
    new_system.start_simulation()