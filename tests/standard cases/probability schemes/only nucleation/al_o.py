from engine import *
import inspect

if __name__ == '__main__':

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

            self.c_automata.cases.first_mp.nucleation_probabilities = utils.NucleationProbabilities(
                Config.PROBABILITIES.PRIMARY,Config.PRODUCTS.PRIMARY)

            self.save_function = None


    source_code = inspect.getsource(NewSystem)
    Config.INITIAL_SCRIPT += source_code
    Config.COMMENT = "This script simulates oxidation with nucleation scheme, no dissolution!"
    new_system = NewSystem()
    new_system.start_simulation()
