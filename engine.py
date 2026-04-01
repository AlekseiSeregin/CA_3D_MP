from cellular_automata import *
from utils import data_base
from tqdm import tqdm
import time
import keyboard
from microstructure import voronoi
import elements
import numpy as np
from diffusion_3d_mp_example import DiffusionEngine as _DiffusionEngine
from workers.worker_pools import WorkerPools



class FunctionBlock:
    def __init__(self):
        self.__func_block = []

    def add_func(self, func):
        self.__func_block.append(func)

    def execute(self):
        [cur_f() for cur_f in self.__func_block]


class SimulationConfigurator:

    """
    TODO: 2. Buffer reserve for cells array to Config!!
          3. DEFAULT_PARAMS in templates, move out or create different script for cases!
          4. Utils away from the element classes
          5. self.cases.reaccumulate_products_no_exclusion() for the oxidant diffusion implements subtitution for dummy function
          in case where only one product.

    Check list for functions:
    fix_init_precip - can be Dummy!
    """

    def __init__(self):
        self.utils = utils.Utils()
        self.utils.generate_param()
        self.cases = utils.CaseRef()
        self.c_automata = CellularAutomata(self.cases, self.utils)

        self.db = data_base.Database()
        self.begin = None
        self.elapsed_time = None

        # setting objects for inward diffusion
        if Config.INWARD_DIFFUSION:
            self.init_inward()
        # setting objects for outward diffusion
        if Config.OUTWARD_DIFFUSION:
            self.init_outward()
        # setting objects for precipitation
        if Config.COMPUTE_PRECIPITATION:
            self.init_product()

        # New 3D shared-memory diffusion engine (optional; legacy path remains when USE_NEW_DIFFUSION_ENGINE is False)
        self.diffusion_engine = None
        self.worker_pools = None
        if Config.INWARD_DIFFUSION or Config.OUTWARD_DIFFUSION:
            n_out = getattr(Config, 'OUTWARD_DIFFUSION_WORKERS', 5)
            n_in = getattr(Config, 'INWARD_DIFFUSION_WORKERS', 5)
            rng = np.random.default_rng()
            # Standalone worker pools for CA + diffusion; pools are not hosted inside DiffusionEngine.
            self.worker_pools = WorkerPools(n_outward_workers=n_out, n_inward_workers=n_in)
            self.c_automata.worker_pools = self.worker_pools
            self.c_automata._ensure_precip_z_states()

            self._diffusion_engine = _DiffusionEngine(n_out, n_in, rng, worker_pools=self.worker_pools)
            self.c_automata.diffusion_engine = self._diffusion_engine

        self.function_block = FunctionBlock()
        self.current_func = None  # must be defined elsewhere
        self.save_function = None  # must be defined elsewhere

        self.termination_command = Config.TERMINATION_COMMAND

    def start_simulation(self):
        try:
            self.__construct_function_block()
            self.__start_execution()
        finally:
            if self._diffusion_engine is not None:
                self._diffusion_engine.close()
                self._diffusion_engine = None
            if self.worker_pools is not None:
                self.worker_pools.close()
                self.worker_pools = None
            self.save_results()
            self.insert_last_it()
            self.db.conn.commit()
            print()
            print("____________________________________________________________")
            print("Simulation was closed at Iteration: ", self.c_automata.iteration)
            print("____________________________________________________________")
            print()

    def __start_execution(self):
        self.begin = time.time()
        for self.c_automata.iteration in tqdm(
            range(Config.N_ITERATIONS),
            desc="Simulation",
            dynamic_ncols=True,
        ):
            if keyboard.is_pressed(self.termination_command):
                break
            self.function_block.execute()
        end = time.time()
        self.elapsed_time = (end - self.begin)
        self.db.insert_time(self.elapsed_time)
        self.db.conn.commit()

    def init_inward(self):
        self.cases.add_oxidant(elements.OxidantElem(Config.OXIDANTS.PRIMARY, self.utils))

        # self.cases.first.oxidant = elements.OxidantElem(Config.OXIDANTS.PRIMARY, self.utils)
        # self.cases.second.oxidant = self.cases.first.oxidant

        # self.cases.first_mp.oxidant_c3d_shm_mdata = getattr(self.cases.first.oxidant, 'c3d_shm_mdata', None)
        # self.cases.second_mp.oxidant_c3d_shm_mdata = self.cases.first_mp.oxidant_c3d_shm_mdata

        # self.cases.third.oxidant = self.cases.first.oxidant
        # self.cases.fourth.oxidant = self.cases.first.oxidant
        # self.cases.fifth.oxidant = self.cases.first.oxidant

        # self.cases.third_mp.oxidant_c3d_shm_mdata = self.cases.first_mp.oxidant_c3d_shm_mdata
        # self.cases.fourth_mp.oxidant_c3d_shm_mdata = self.cases.first_mp.oxidant_c3d_shm_mdata
        # self.cases.fifth_mp.oxidant_c3d_shm_mdata = self.cases.first_mp.oxidant_c3d_shm_mdata

        # # ---------------------------------------------------
        # if Config.OXIDANTS.SECONDARY_EXISTENCE:
        #     self.cases.third.oxidant = elements.OxidantElem(Config.OXIDANTS.SECONDARY, self.utils)
        #     self.cases.fourth.oxidant = self.cases.third.oxidant
        #     self.cases.third_mp.oxidant_c3d_shm_mdata = getattr(self.cases.third.oxidant, 'c3d_shm_mdata', None)
        #     self.cases.fourth_mp.oxidant_c3d_shm_mdata = self.cases.third_mp.oxidant_c3d_shm_mdata

    def init_outward(self):
        self.cases.add_active(elements.ActiveElem(Config.ACTIVES.PRIMARY))

        # self.cases.first.active = elements.ActiveElem(Config.ACTIVES.PRIMARY)
        # self.cases.third.active = self.cases.first.active

        # # ---------------------------------------------------
        # # c3d (with new diffusion, active exposes count buffer as c3d_shm_mdata for nucleation)
        # self.cases.first_mp.active_c3d_shm_mdata = getattr(self.cases.first.active, 'c3d_shm_mdata', None)
        # self.cases.third_mp.active_c3d_shm_mdata = self.cases.first_mp.active_c3d_shm_mdata
        # self.cases.fifth_mp.active_c3d_shm_mdata = self.cases.first_mp.active_c3d_shm_mdata  # JUST FOR SHAPE!!!
        # # cells/dirs (legacy flat arrays; None when using USE_NEW_DIFFUSION_ENGINE)
        # self.cases.first_mp.active_cells_shm_mdata = getattr(self.cases.first.active, 'cells_shm_mdata', None)
        # self.cases.third_mp.active_cells_shm_mdata = self.cases.first_mp.active_cells_shm_mdata
        # self.cases.first_mp.active_dirs_shm_mdata = getattr(self.cases.first.active, 'dirs_shm_mdata', None)
        # self.cases.third_mp.active_dirs_shm_mdata = self.cases.first_mp.active_dirs_shm_mdata

        # # ---------------------------------------------------
        # if Config.ACTIVES.SECONDARY_EXISTENCE:
        #     self.cases.second.active = elements.ActiveElem(Config.ACTIVES.SECONDARY)
        #     self.cases.fourth.active = self.cases.second.active
        #     # ---------------------------------------------------
        #     self.cases.second_mp.active_c3d_shm_mdata = getattr(self.cases.second.active, 'c3d_shm_mdata', None)
        #     self.cases.fourth_mp.active_c3d_shm_mdata = self.cases.second_mp.active_c3d_shm_mdata
        #     self.cases.second_mp.active_cells_shm_mdata = getattr(self.cases.second.active, 'cells_shm_mdata', None)
        #     self.cases.fourth_mp.active_cells_shm_mdata = self.cases.second_mp.active_cells_shm_mdata
        #     self.cases.second_mp.active_dirs_shm_mdata = getattr(self.cases.second.active, 'dirs_shm_mdata', None)
        #     self.cases.fourth_mp.active_dirs_shm_mdata = self.cases.second_mp.active_dirs_shm_mdata

    @staticmethod
    def _normalize_component_names(components):
        if components is None:
            return []
        return [str(comp).strip() for comp in components if str(comp).strip() and str(comp).strip().lower() != "none"]

    def _build_species_maps(self):
        oxidants_by_element = {}
        actives_by_element = {}
        for oxidant in self.cases.all_oxidants:
            oxidants_by_element[oxidant.elem_name] = oxidant
        for active in self.cases.all_actives:
            actives_by_element[active.elem_name] = active
        return oxidants_by_element, actives_by_element

    def _get_configured_product_definitions(self):
        product_groups = getattr(Config, "PRODUCTS", None)
        if product_groups is None:
            return []
        definitions = []
        declaration_order = []
        for key in ("PRIMARY", "SECONDARY", "TERNARY", "QUATERNARY", "QUINT"):
            if hasattr(product_groups, key):
                declaration_order.append(key)
        for key in product_groups.__dict__.keys():
            if key.startswith("_"):
                continue
            if key not in declaration_order:
                declaration_order.append(key)

        for idx, key in enumerate(declaration_order):
            cfg = getattr(product_groups, key, None)
            if cfg is None:
                continue
            thr_in = int(getattr(cfg, "THRESHOLD_INWARD", 0))
            thr_out = int(getattr(cfg, "THRESHOLD_OUTWARD", 0))
            if thr_in <= 0 or thr_out <= 0:
                continue
            priority = getattr(cfg, "PRIORITY", None)
            if priority is None:
                priority = idx + 1
            components = self._normalize_component_names(getattr(cfg, "COMPONENTS", []))
            definitions.append({
                "key": key,
                "cfg": cfg,
                "priority": int(priority),
                "components": components,
                "element": str(getattr(cfg, "ELEMENT", key)),
                "decl_idx": idx,
            })
        definitions.sort(key=lambda item: (item["priority"], item["decl_idx"]))
        return definitions

    def _resolve_reactants_from_components(self, components):
        oxidants_by_element, actives_by_element = self._build_species_maps()
        oxidant = None
        active = None
        for comp in components:
            if oxidant is None and comp in oxidants_by_element:
                oxidant = oxidants_by_element[comp]
            if active is None and comp in actives_by_element:
                active = actives_by_element[comp]
            if oxidant is not None and active is not None:
                break
        return oxidant, active

    def _apply_case_reactants_from_components(self, case, case_mp, product_def):
        oxidant, active = self._resolve_reactants_from_components(product_def["components"])
        case.oxidant = oxidant
        case_mp.oxidant_c3d_shm_mdata = oxidant.c3d_shm_mdata
        case.active = active
        case_mp.active_c3d_shm_mdata = active.c3d_shm_mdata
        case_mp.active_cells_shm_mdata = active.cells_shm_mdata
        case_mp.active_dirs_shm_mdata = active.dirs_shm_mdata

    def init_product(self):
        # c3d_init
        tmp = np.zeros((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), dtype=np.ubyte)
        self.cases.precip_3d_init_shm = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.cases.precip_3d_init = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.cases.precip_3d_init_shm.buf)
        np.copyto(self.cases.precip_3d_init, tmp)
        self.cases.precip_3d_init_shm_mdata = SharedMetaData(self.cases.precip_3d_init_shm.name, tmp.shape, tmp.dtype)

        # accumulated products
        # tmp = np.zeros((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), dtype=np.ubyte)
        # self.cases.accumulated_products_shm = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        # self.cases.accumulated_products = np.ndarray(tmp.shape, dtype=tmp.dtype,
        #                                              buffer=self.cases.accumulated_products_shm.buf)
        # np.copyto(self.cases.accumulated_products, tmp)
        # self.cases.accumulated_products_shm_mdata = SharedMetaData(self.cases.accumulated_products_shm.name, tmp.shape,
        #                                                            tmp.dtype)
        # Unified product state map: [owner_id, count] in one shared array.
        # state[0]: owner id (0 empty, 1..255 phase id)
        # state[1]: occupancy count in the owned phase cell (0..oxidation_number)
        state_tmp = np.zeros((2, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), dtype=np.uint8)
        self.cases.product_state_shm = shared_memory.SharedMemory(create=True, size=state_tmp.nbytes)
        self.cases.product_state = np.ndarray(state_tmp.shape, dtype=state_tmp.dtype, buffer=self.cases.product_state_shm.buf)
        np.copyto(self.cases.product_state, state_tmp)
        self.cases.product_state_shm_mdata = SharedMetaData(self.cases.product_state_shm.name, state_tmp.shape, state_tmp.dtype)

        product_defs = self._get_configured_product_definitions()
        if len(product_defs) == 0:
            return
        while len(self.cases.product_cases) < len(product_defs):
            self.cases.add_case()
        case_slots = list(zip(self.cases.product_cases, self.cases.product_cases_mp))

        stage_sequence = []
        self.cases.product_cases_by_key = {}
        self.cases.product_cases_by_phase_id = {}
        for idx, (product_def, (case, case_mp)) in enumerate(zip(product_defs, case_slots), start=1):
            self.init_case(
                case,
                case_mp,
                product_def["cfg"],
                phase_id=idx,
                product_key=product_def["key"],
                product_element=product_def["element"],
                components=product_def["components"],
                stage_priority=product_def["priority"],
            )
            self._apply_case_reactants_from_components(case, case_mp, product_def)
            case.is_active = True
            case_mp.is_active = True
            stage_sequence.append((case, case_mp))
            self.cases.product_cases_by_key[case_mp.product_key] = (case, case_mp)
            self.cases.product_cases_by_phase_id[case_mp.product_phase_id] = (case, case_mp)

        self.c_automata.product_stage_sequence = stage_sequence
        self.c_automata.cur_case, self.c_automata.cur_case_mp = stage_sequence[0]

    def init_case(
        self,
        case,
        case_mp,
        product_config,
        phase_id=1,
        product_key=None,
        product_element=None,
        components=(),
        stage_priority=0,
    ):
        # product data is owned by case (not by elements.Product)
        case.product_oxidation_number = int(product_config.OXIDATION_NUMBER)
        
        case_mp.oxidation_number = case.product_oxidation_number
        case_mp.threshold_inward = product_config.THRESHOLD_INWARD
        case_mp.threshold_outward = product_config.THRESHOLD_OUTWARD
        case_mp.product_phase_id = int(phase_id)
        case_mp.product_state_shm_mdata = self.cases.product_state_shm_mdata
        mode = resolve_nucleation_mode(
            getattr(Config, "NUCLEATION_MODE", None),
            getattr(Config, "USE_SIMPLE_NUCLEATION", False),
        )
        case_mp.nucleation_mode = mode
        case_mp.nucleation_kernel_runner = get_nucleation_kernel_runner(mode)
        case_mp.product_key = product_key
        case_mp.product_element = product_element
        case_mp.product_components = tuple(components)
        case_mp.stage_priority = int(stage_priority)
        case.fix_init_precip_func_ref = self.c_automata.fix_init_precip_int


    def save_results(self):
        # With USE_NEW_DIFFUSION_ENGINE, oxidant.cells and active.get_cells_coords() read from the 3D diffusion grid (same DB format).
        if Config.STRIDE > Config.N_ITERATIONS:
            self.cases.first.active.transform_to_descards()
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.cases.second.active.transform_to_descards()
        if Config.INWARD_DIFFUSION:
            self.db.insert_particle_data("primary_oxidant", self.c_automata.iteration, self.cases.first.oxidant.cells)
            if Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_oxidant", self.c_automata.iteration, self.cases.second.oxidant.cells)
        if Config.OUTWARD_DIFFUSION:
            self.db.insert_particle_data("primary_active", self.c_automata.iteration, self.cases.first.active.get_cells_coords())
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_active", self.c_automata.iteration, self.cases.second.active.get_cells_coords())
        if Config.COMPUTE_PRECIPITATION:
            self.db.insert_particle_data("primary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.first, self.cases.first_mp))
            if Config.ACTIVES.SECONDARY_EXISTENCE and Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.second, self.cases.second_mp))
                self.db.insert_particle_data("ternary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.third, self.cases.third_mp))
                self.db.insert_particle_data("quaternary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.fourth, self.cases.fourth_mp))
            elif Config.ACTIVES.SECONDARY_EXISTENCE and not Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.second, self.cases.second_mp))
        if Config.STRIDE > Config.N_ITERATIONS:
            self.cases.first.active.transform_to_3d(self.c_automata.curr_max_furthest)
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.cases.second.active.transform_to_3d(self.c_automata.curr_max_furthest)

    def _get_product_save_coords(self, case, case_mp):
        """
        Return product coordinates as (3, n) with rows [z, y, x] for DB insert.
        Uses unified product_state (owner + count) in owner modes; falls back to legacy product arrays.
        """
        pid = int(getattr(case_mp, "product_phase_id", 0))
        state = getattr(self.cases, "product_state", None)
        if state is None or pid <= 0:
            return case.product.transform_c3d()

        owner = state[0]
        count = state[1]
        occupied = np.array(np.nonzero(owner == np.uint8(pid)), dtype=np.short)
        if occupied.shape[1] == 0:
            return np.empty((3, 0), dtype=np.short)
        coords = np.array([occupied[2, :], occupied[1, :], occupied[0, :]], dtype=np.short)
        counts = count[occupied[0], occupied[1], occupied[2]]
        return np.array(np.repeat(coords, counts, axis=1), dtype=np.short)

    def save_microstructure(self, microstructure):
        self.db.save_pickled_microstructure(microstructure)

    def save_results_custom(self):
        if Config.STRIDE > Config.N_ITERATIONS:
            self.cases.first.active.transform_to_descards()
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.cases.second.active.transform_to_descards()

        self.db.insert_particle_data("primary_oxidant", self.c_automata.iteration, self.cases.first.oxidant.cells)

        self.db.insert_particle_data("primary_active", self.c_automata.iteration, self.cases.first.active.get_cells_coords())
        self.db.insert_particle_data("secondary_active", self.c_automata.iteration, self.cases.second.active.get_cells_coords())

        self.db.insert_particle_data("primary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.first, self.cases.first_mp))
        self.db.insert_particle_data("secondary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.second, self.cases.second_mp))
        self.db.insert_particle_data("ternary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.third, self.cases.third_mp))
        self.db.insert_particle_data("quaternary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.fourth, self.cases.fourth_mp))
        self.db.insert_particle_data("quint_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.fifth, self.cases.fifth_mp))

        if Config.STRIDE > Config.N_ITERATIONS:
            self.cases.first.active.transform_to_3d(self.c_automata.curr_max_furthest)
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.cases.second.active.transform_to_3d(self.c_automata.curr_max_furthest)

    def calc_precipitation_front_only_cells(self):
        """
        Calculating a position of a precipitation front, considering only cells concentrations without any scaling!
        As a boundary a product fraction of 50% is used.
        """
        product = np.array([np.sum(self.c_automata.cur_case.product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.c_automata.cells_per_axis)], dtype=np.uint32)
        product = product / (self.c_automata.cells_per_axis ** 2)

        if self.c_automata.iteration % Config.STRIDE == 0:
            self.c_automata.record_prod_per_layer(product.shape[0]-1, product, np.zeros(product.shape))

        threshold = Config.ACTIVES.PRIMARY.CELLS_CONCENTRATION
        for rev_index, precip_conc in enumerate(np.flip(product)):
            if precip_conc > threshold / 2:
                position = (len(product) - 1 - rev_index) * Config.SIZE * 10 ** 6 \
                           / self.c_automata.cells_per_axis
                sqr_time = ((self.c_automata.iteration + 1) * Config.SIM_TIME / (self.c_automata.n_iter * 3600)) ** (1 / 2)
                self.db.insert_precipitation_front(sqr_time, position, "p")
                break

    def terminate_workers(self):
        self.c_automata.pool.close()
        self.c_automata.pool.join()
        print("TERMINATED PROPERLY!")

    def unlink(self):
        self.cases.close_shms()
        print("UNLINKED PROPERLY!")

    def save_results_only_prod_prime(self):
        self.db.insert_particle_data("primary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.first, self.cases.first_mp))

    def save_results_only_prod(self):
        self.db.insert_particle_data("primary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.first, self.cases.first_mp))
        self.db.insert_particle_data("secondary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.second, self.cases.second_mp))
        self.db.insert_particle_data("ternary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.third, self.cases.third_mp))
        self.db.insert_particle_data("quaternary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.fourth, self.cases.fourth_mp))
        self.db.insert_particle_data("quint_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.fifth, self.cases.fifth_mp))

    def save_results_only_prod_secondary(self):
        self.db.insert_particle_data("secondary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.second, self.cases.second_mp))

    def save_results_prod_and_inw(self):
        self.db.insert_particle_data("primary_product", self.c_automata.iteration, self._get_product_save_coords(self.cases.first, self.cases.first_mp))
        self.db.insert_particle_data("primary_oxidant", self.c_automata.iteration, self.cases.first.oxidant.cells)

    def save_results_only_inw(self):
        self.db.insert_particle_data("primary_oxidant", self.c_automata.iteration, self.cases.first.oxidant.cells)

    def insert_last_it(self):
        self.db.insert_last_iteration(self.c_automata.iteration)

    def __construct_function_block(self):
        """The execution order is: Nucleation -> Dissolution -> Inward Diffusion -> Outward Diffusion -> Save.
        Depending on the initial conditions some steps can be skipped, the execution sequence will be adjusted"""
        # Nucleation
        if Config.COMPUTE_PRECIPITATION and self.c_automata.precip_func is not None:
            self.function_block.add_func(self.c_automata.precip_func)
        # Dissolution
        if Config.DECOMPOSE_PRECIPITATIONS and self.c_automata.decomposition is not None:
            self.function_block.add_func(self.c_automata.decomposition)
        # Diffusion
        if Config.INWARD_DIFFUSION or Config.OUTWARD_DIFFUSION:
            self.function_block.add_func(self.c_automata.diffuse_all)
        # Save
        if Config.SAVE_WHOLE and self.save_function is not None:
            self.function_block.add_func(self.save_function)
