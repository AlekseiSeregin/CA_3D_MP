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
            self.c_automata.cases.first_mp.check_intersection = ci_single_no_growth

            self.cases.first.fix_init_precip_func_ref = self.c_automata.fix_init_precip_dummy


    source_code = inspect.getsource(NewSystem)
    Config.INITIAL_SCRIPT += source_code
    Config.COMMENT = "This script simulates the general Wagner case with some permeability of active element."
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