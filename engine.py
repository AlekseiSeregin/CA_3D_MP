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
from types import SimpleNamespace



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
            self.c_automata.ensure_jmatpro_pool()
            self.c_automata._ensure_precip_z_states()

            self._diffusion_engine = _DiffusionEngine(
                n_out, n_in, rng,
                worker_pools=self.worker_pools,
                product_state_shm_mdata=getattr(self.cases, "product_state_shm_mdata", None),
            )
            self.c_automata.diffusion_engine = self._diffusion_engine

        self.function_block = FunctionBlock()
        self.current_func = None  # must be defined elsewhere
        self.save_function = None  # must be defined elsewhere

        self.termination_command = Config.TERMINATION_COMMAND
        self.c_automata._init_jmatpro_product_scan_cache()
        self.print_config()

    def print_config(self):
        all_oxidants = Config.OXIDANTS[0]["N_PER_PAGE"] * Config.OXIDANTS[0]["MOLES_PER_CELL"]
        all_actives = Config.ACTIVES[0]["N_PER_PAGE"] * Config.ACTIVES[0]["MOLES_PER_CELL"]
        all_matrix = Config.N_CELLS_PER_AXIS ** 2 * Config.MATRIX.MOLES_PER_CELL
        all_matrix = all_matrix - Config.ACTIVES[0]["N_PER_PAGE"] * Config.ACTIVES[0]["MOLES_PER_CELL"] * Config.ACTIVES[0]["T"]

        all_oxidants_m = all_oxidants * Config.OXIDANTS[0]["MOLAR_MASS"]
        all_actives_m = all_actives * Config.ACTIVES[0]["MOLAR_MASS"]
        all_matrix_m = all_matrix * Config.MATRIX.MOLAR_MASS
        all_total_m = all_oxidants_m + all_actives_m + all_matrix_m

        c_all_oxidants_m = all_oxidants_m / all_total_m
        c_all_actives_m = all_actives_m / all_total_m
        c_all_matrix_m = all_matrix_m / all_total_m
        c_only_actives_m = all_actives_m / (all_actives_m + all_matrix_m)

        print(f"c_all_oxidants_m: {c_all_oxidants_m}")
        print(f"c_all_actives_m: {c_all_actives_m}")
        print(f"c_all_matrix_m: {c_all_matrix_m}")
        print(f"c_only_actives_m: {c_only_actives_m}")

        c_all_oxidants = all_oxidants / (all_oxidants + all_actives + all_matrix)
        c_all_actives = all_actives / (all_oxidants + all_actives + all_matrix)
        c_all_matrix = all_matrix / (all_oxidants + all_actives + all_matrix)
        c_only_actives = all_actives / (all_actives + all_matrix)

        print(f"c_all_oxidants: {c_all_oxidants}")
        print(f"c_all_actives: {c_all_actives}")
        print(f"c_all_matrix: {c_all_matrix}")
        print(f"c_only_actives: {c_only_actives}")

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
            self.db.insert_product_plane0_tracking(self.c_automata.product_plane_tracking)
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
        oxidants = getattr(Config, "OXIDANTS_RUNTIME", [])
        for ox_cfg in oxidants:
            self.cases.add_oxidant(elements.OxidantElem(ox_cfg, self.utils))

    def init_outward(self):
        actives = getattr(Config, "ACTIVES_RUNTIME", [])
        for act_cfg in actives:
            self.cases.add_active(elements.ActiveElem(act_cfg))

    @staticmethod
    def _normalize_component_names(components):
        if components is None:
            return []
        return [str(comp).strip() for comp in components if str(comp).strip() and str(comp).strip().lower() != "none"]

    @staticmethod
    def _make_product_cfg(raw_def, idx):
        if not isinstance(raw_def, dict):
            raise ValueError(f"PRODUCTS[{idx}] must be a dictionary.")

        key = str(raw_def.get("key", f"product_{idx + 1}")).strip().lower()
        element = str(raw_def.get("element", "")).strip()
        jm_identifier = str(raw_def.get("jm_identifier", "")).strip()
        priority = int(raw_def.get("priority", idx + 1))
        stoich_raw = raw_def.get("stoich", {})
        outward_raw = raw_def.get("outward_element", "")
        inward_raw = raw_def.get("inward_element", "")
        outward_element = "" if outward_raw is None else str(outward_raw).strip()
        inward_element = "" if inward_raw is None else str(inward_raw).strip()

        if not key:
            raise ValueError(f"PRODUCTS[{idx}] has empty key.")
        if not element:
            raise ValueError(f"PRODUCTS[{idx}] has empty element.")
        if not jm_identifier:
            raise ValueError(f"PRODUCTS[{idx}] has empty jm_identifier.")
        if not isinstance(stoich_raw, dict) or len(stoich_raw) == 0:
            raise ValueError(f"PRODUCTS[{idx}] must define non-empty stoich dictionary.")
        if not inward_element:
            raise ValueError(f"PRODUCTS[{idx}] must define non-empty inward_element.")

        stoich = {}
        for elem, val in stoich_raw.items():
            ev = str(elem).strip()
            if not ev:
                continue
            iv = int(val)
            if iv <= 0:
                raise ValueError(f"PRODUCTS[{idx}] stoich[{ev}] must be > 0.")
            stoich[ev] = iv
        if len(stoich) == 0:
            raise ValueError(f"PRODUCTS[{idx}] has no valid stoichiometric entries.")

        comp_set = set(stoich.keys())
        if outward_element and outward_element not in comp_set:
            raise ValueError(f"PRODUCTS[{idx}] outward_element must be part of stoich keys.")
        if inward_element not in comp_set:
            raise ValueError(f"PRODUCTS[{idx}] inward_element must be part of stoich keys.")
        if outward_element and outward_element == inward_element:
            raise ValueError(f"PRODUCTS[{idx}] inward_element and outward_element must not be the same.")

        product_cfg = SimpleNamespace()
        product_cfg.KEY = key
        product_cfg.ELEMENT = element
        product_cfg.JM_IDENTIFIER = jm_identifier
        product_cfg.PRIORITY = priority
        product_cfg.STOICH = stoich
        product_cfg.OUTWARD_ELEMENT = outward_element
        product_cfg.INWARD_ELEMENT = inward_element
        product_cfg.COMPONENTS = list(stoich.keys())
        thr_out = int(raw_def.get("threshold_outward", raw_def.get("THRESHOLD_OUTWARD", 0)))
        thr_in = int(raw_def.get("threshold_inward", raw_def.get("THRESHOLD_INWARD", 0)))
        if thr_in <= 0:
            raise ValueError(f"PRODUCTS[{idx}] must define positive threshold_inward.")
        if outward_element:
            if thr_out <= 0:
                raise ValueError(f"PRODUCTS[{idx}] with outward_element must define positive threshold_outward.")
        else:
            if thr_out != 0:
                raise ValueError(f"PRODUCTS[{idx}] without outward_element must set threshold_outward = 0.")
        product_cfg.THRESHOLD_OUTWARD = thr_out
        product_cfg.THRESHOLD_INWARD = thr_in
        product_cfg.MASS_PER_CELL = float(raw_def.get("MASS_PER_CELL", 0.0))
        product_cfg.MOLES_PER_CELL = float(raw_def.get("MOLES_PER_CELL", 0.0))
        product_cfg.MATRIX_MASS_PER_CELL = float(raw_def.get("MATRIX_MASS_PER_CELL", 0.0))
        product_cfg.MATRIX_MOLES_PER_CELL = float(raw_def.get("MATRIX_MOLES_PER_CELL", 0.0))
        product_cfg.CONSTITUTION = str(raw_def.get("CONSTITUTION", "+".join(product_cfg.COMPONENTS)))
        product_cfg.OXIDATION_NUMBER = int(raw_def.get("OXIDATION_NUMBER", 1))
        product_cfg.LIND_FLAT_ARRAY = int(raw_def.get("LIND_FLAT_ARRAY", 6))
        product_cfg.PHASE_FRACTION_LIMIT = float(raw_def.get("PHASE_FRACTION_LIMIT", Config.PHASE_FRACTION_LIMIT))
        dissolution_ratio = float(raw_def.get("dissolution_time_ratio", raw_def.get("DISSOLUTION_TIME_RATIO", 0.0)))
        if dissolution_ratio < 0.0:
            raise ValueError(f"PRODUCTS[{idx}] dissolution_time_ratio must be >= 0.")
        product_cfg.DISSOLUTION_TIME_RATIO = dissolution_ratio
        probs_raw = raw_def.get("probabilities", None)
        if probs_raw is None:
            raise ValueError(f"PRODUCTS[{idx}] must define 'probabilities'.")
        product_cfg.PROBABILITIES = SimulationConfigurator._make_probabilities_cfg(probs_raw, idx)
        return product_cfg

    @staticmethod
    def _make_probabilities_cfg(probs_raw, idx):
        if not isinstance(probs_raw, dict):
            raise ValueError(f"PRODUCTS[{idx}].probabilities must be a dictionary.")
        p_cfg = SimpleNamespace()
        required = [
            "p0", "p0_f", "p0_A_const", "p0_B_const",
            "p1", "p1_f", "p1_A_const", "p1_B_const",
            "global_A", "global_B", "global_B_f", "max_neigh_numb", "nucl_adapt_function",
            "p0_d", "p0_d_f", "p0_d_A_const", "p0_d_B_const",
            "p1_d", "p1_d_f", "p1_d_A_const", "p1_d_B_const",
            "p6_d", "p6_d_f", "p6_d_A_const", "p6_d_B_const",
            "global_d_A", "global_d_B", "global_d_B_f", "bsf", "dissol_adapt_function",
        ]
        missing = [k for k in required if k not in probs_raw]
        if missing:
            raise ValueError(f"PRODUCTS[{idx}].probabilities missing keys: {missing}")
        for k, v in probs_raw.items():
            setattr(p_cfg, k, v)
        return p_cfg

    def _build_species_maps(self):
        oxidants_by_element = {}
        actives_by_element = {}
        for oxidant in self.cases.all_oxidants:
            oxidants_by_element[oxidant.elem_name] = oxidant
        for active in self.cases.all_actives:
            actives_by_element[active.elem_name] = active
        return oxidants_by_element, actives_by_element

    def _get_configured_product_definitions(self):
        products = getattr(Config, "PRODUCTS", None)
        if products is None:
            return []
        if not isinstance(products, (list, tuple)):
            raise ValueError("Config.PRODUCTS must be a list of product dictionaries.")

        definitions = []
        seen_keys = set()
        for idx, raw_def in enumerate(products):
            cfg = self._make_product_cfg(raw_def, idx)
            if cfg.KEY in seen_keys:
                raise ValueError(f"Duplicate product key '{cfg.KEY}' in Config.PRODUCTS.")
            seen_keys.add(cfg.KEY)
            definitions.append({
                "key": cfg.KEY,
                "cfg": cfg,
                "priority": int(cfg.PRIORITY),
                "components": list(cfg.COMPONENTS),
                "element": str(cfg.ELEMENT),
                "inward_element": str(cfg.INWARD_ELEMENT),
                "outward_element": str(cfg.OUTWARD_ELEMENT),
                "decl_idx": idx,
            })
        definitions.sort(key=lambda item: (item["priority"], item["decl_idx"]))
        return definitions

    def _apply_case_reactants_from_components(self, case, case_mp, product_def):
        oxidants_by_element, actives_by_element = self._build_species_maps()
        inward_element = product_def["inward_element"]
        outward_element = product_def["outward_element"]
        missing_inward = inward_element not in oxidants_by_element
        missing_outward = bool(outward_element) and (outward_element not in actives_by_element)
        if missing_inward or missing_outward:
            raise ValueError(
                f"Product '{product_def['element']}' is missing configured reactants. "
                f"Missing inward oxidant: {inward_element if missing_inward else None}; "
                f"missing outward active: {outward_element if missing_outward else None}"
            )
        oxidant = oxidants_by_element[inward_element]
        active = actives_by_element.get(outward_element) if outward_element else None
        case.oxidant = oxidant
        case_mp.oxidant_c3d_shm_mdata = oxidant.c3d_shm_mdata
        case.active = active
        case_mp.active_c3d_shm_mdata = active.c3d_shm_mdata if active is not None else None
        case_mp.active_cells_shm_mdata = active.cells_shm_mdata if active is not None else None
        case_mp.active_dirs_shm_mdata = active.dirs_shm_mdata if active is not None else None

    def init_product(self):
        # c3d_init
        tmp = np.zeros(
            (Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS),
            dtype=np.uint16,
        )
        self.cases.precip_3d_init_shm = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.cases.precip_3d_init = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.cases.precip_3d_init_shm.buf)
        np.copyto(self.cases.precip_3d_init, tmp)
        self.cases.precip_3d_init_shm_mdata = SharedMetaData(self.cases.precip_3d_init_shm.name, tmp.shape, tmp.dtype)

        state_tmp = np.zeros(
            (2, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS),
            dtype=np.uint16,
        )
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
        case_mp.dissolution_time_ratio = float(getattr(product_config, "DISSOLUTION_TIME_RATIO", 0.0))
        case_mp.dissolution_n_iterations = int(round(float(Config.N_ITERATIONS) * case_mp.dissolution_time_ratio))
        case_mp.dissolution_counter = np.zeros(Config.N_CELLS_PER_AXIS, dtype=np.int32)
        case_mp.dissolution_count_activated = np.zeros(Config.N_CELLS_PER_AXIS, dtype=bool)
        # Keep case-level mirrors for compatibility with code paths using case instead of case_mp.
        case.dissolution_time_ratio = case_mp.dissolution_time_ratio
        case.dissolution_n_iterations = case_mp.dissolution_n_iterations
        case.dissolution_counter = case_mp.dissolution_counter
        case.dissolution_count_activated = case_mp.dissolution_count_activated
        case_mp.product_phase_id = int(phase_id)
        case_mp.product_state_shm_mdata = self.cases.product_state_shm_mdata
        mode = resolve_nucleation_mode(
            getattr(Config, "NUCLEATION_MODE", None),
            getattr(Config, "USE_SIMPLE_NUCLEATION", False),
        )
        no_outward = (not bool(str(getattr(product_config, "OUTWARD_ELEMENT", "")).strip())) or int(getattr(product_config, "THRESHOLD_OUTWARD", 0)) <= 0
        case_mp.no_outward_nucleation = bool(no_outward)
        case_mp.severe_dissolution_indexes = []
        case_mp.nucleation_mode = mode
        fold = bool(getattr(Config, "NUCLEATION_APPLY_FOLD", False))
        case_mp.nucleation_apply_fold = fold
        if fold:
            if mode not in ("legacy_prob_owner", "stoich_prob_owner"):
                raise ValueError(
                    "NUCLEATION_APPLY_FOLD=True requires NUCLEATION_MODE to be "
                    "'legacy_prob_owner' or 'stoich_prob_owner'; got %r" % (mode,)
                )
            case_mp.nucleation_kernel_runner = get_nucleation_kernel_runner_fold(
                mode, no_outward=case_mp.no_outward_nucleation
            )
        else:
            case_mp.nucleation_kernel_runner = get_nucleation_kernel_runner(
                mode, no_outward=case_mp.no_outward_nucleation
            )
        case_mp.product_key = product_key
        case_mp.product_cfg = product_config
        case_mp.matrix_moles_per_cell = float(getattr(product_config, "MATRIX_MOLES_PER_CELL", 0.0))
        case_mp.product_element = product_element
        case_mp.product_components = tuple(components)
        case_mp.stage_priority = int(stage_priority)
        case_mp.jm_identifier = str(getattr(product_config, "JM_IDENTIFIER", ""))
        case_mp.outward_element = str(getattr(product_config, "OUTWARD_ELEMENT", ""))
        stoich = getattr(product_config, "STOICH", {}) or {}
        nu_sum = float(sum(float(v) for v in stoich.values())) if len(stoich) > 0 else 0.0
        if nu_sum > 0.0:
            case_mp.stoich_frac_items = tuple(
                (str(elem), float(val) / nu_sum) for elem, val in stoich.items() if float(val) > 0.0
            )
        else:
            case_mp.stoich_frac_items = ()
        case.fix_init_precip_func_ref = self.c_automata.fix_init_precip_int
        case_mp.precip_3d_init_shm_mdata = self.cases.precip_3d_init_shm_mdata
        case_mp.nucleation_probabilities = utils.NucleationProbabilities(
            product_config.PROBABILITIES,
            product_config
        )
        # Keep both references populated for compatibility with callers that still read from case.
        case_mp.dissolution_probabilities = utils.DissolutionProbabilities(
            product_config.PROBABILITIES,
        )
        case.dissolution_probabilities = case_mp.dissolution_probabilities

    def save_results(self):
        if Config.INWARD_DIFFUSION:
            for oxidant in self.cases.all_oxidants:
                self.db.insert_particle_data(str(oxidant.elem_name), self.c_automata.iteration, oxidant.cells)
        # if Config.OUTWARD_DIFFUSION:
        #     for active in self.cases.all_actives:
        #         self.db.insert_particle_data(str(active.elem_name), self.c_automata.iteration, active.get_cells_coords())
        if Config.COMPUTE_PRECIPITATION:
            for case, case_mp in self.cases.product_case_pairs:
                if case_mp.product_phase_id > 0:
                    self.db.insert_particle_data(
                        str(case_mp.product_element),
                        self.c_automata.iteration,
                        self._get_product_save_coords(case, case_mp),
                    )

    def save_results_product_only(self):
        for case, case_mp in self.cases.product_case_pairs:
            if case_mp.product_phase_id > 0:
                self.db.insert_particle_data(
                    str(case_mp.product_element),
                    self.c_automata.iteration,
                    self._get_product_save_coords(case, case_mp),
                )

    def save_results_inward_only(self):
        if Config.INWARD_DIFFUSION:
            for oxidant in self.cases.all_oxidants:
                self.db.insert_particle_data(
                    str(oxidant.elem_name),
                    self.c_automata.iteration,
                    oxidant.cells,
                )

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

        actives = getattr(Config, "ACTIVES", [])
        threshold = float(actives[0]["cells_concentration"]) if isinstance(actives, list) and len(actives) > 0 else 0.0
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
