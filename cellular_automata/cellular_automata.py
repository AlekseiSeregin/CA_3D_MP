import os
import copy
import numpy as np
from types import SimpleNamespace
import utils
from multiprocessing import shared_memory
from .nes_for_mp import *
from .dissolution_functions import (
    dissolution_subblock_worker,
    dissolution_subblock_worker_blockmask,
    get_block_patterns_from_aggregated,
    _views_from_segment_dissol,
)
from utils.numba_functions import (
    product_counts_upto_bound_from_state,
    product_counts_at_indexes_from_state,
    product_counts_blocks_ignited_from_state,
    severe_planes_clear_product_release,
    severe_blocks_clear_product_release,
)
from thermodynamics import *
from configuration import Config
from diffusion_3d_mp_example import _partition_domain_z, _partition_gap_z_parallel, _DIRS_6_PACKED
from .nucleation_functions import (
    precip_step_subblock_worker,
)
from .neigh_indexes import ind_formation


class CellularAutomata:
    @staticmethod
    def _configured_products():
        products = getattr(Config, "PRODUCTS", [])
        return products if isinstance(products, (list, tuple)) else []

    def __init__(self, cases, utils_inst):
        self.utils = utils_inst
        self.cases = cases
        self.cur_case = None
        self.cur_case_mp = None

        # simulated space parameters
        self.cells_per_axis = Config.N_CELLS_PER_AXIS
        # JMatPro / precip block geometry from Config; fixed for the run.
        self._jmatpro_block_params = self._compute_jmatpro_block_params(int(self.cells_per_axis))
        _bx, _by, _bz = (
            int(self._jmatpro_block_params[3]),
            int(self._jmatpro_block_params[4]),
            int(self._jmatpro_block_params[5]),
        )
        self._jmatpro_n_blocks = _bx * _by * _bz
        # self._init_jmatpro_product_scan_cache()
        self.cells_per_page = self.cells_per_axis ** 2
        self.matrix_moles_per_page = self.cells_per_page * Config.MATRIX.MOLES_PER_CELL
        self.n_iter = Config.N_ITERATIONS
        self.iteration = None
        self.curr_max_furthest = 0
        self.furthest_index = 0
        self.ioz_bound = 0
        self._oxid_cfg_map = {}
        self._act_cfg_map = {}
        self._act_eq_matrix_map = {}
        self._act_t_map = {}
        self._refresh_elem_cfg_cache()

        # New 3D diffusion engine (set by engine when USE_NEW_DIFFUSION_ENGINE is True)
        self.diffusion_engine = None
        # Standalone mp.Pool instances for nucleation/dissolution (set by engine)
        self.worker_pools = None
        # z-partition for precip subblock (computed once on first use, never changes)
        self._precip_z_ranges = None
        self._precip_gap_z_groups = None
        self._precip_z_blocks = None
        self._precip_partition_sig = None
        # Negative partition (swap blocks and gaps) for alternating artefact reduction.
        self._precip_z_ranges_neg = None
        self._precip_gap_z_groups_neg = None
        self._precip_z_blocks_neg = None

        # functions
        self.precip_func = None  # must be defined elsewhere
        self.get_combi_ind = None  # must be defined elsewhere
        self.precip_step = None  # must be defined elsewhere
        self.get_cur_ioz_bound = None  # must be defined elsewhere
        self.get_cur_dissol_ioz_bound = None  # must be defined elsewhere
        self.check_intersection = None  # must be defined elsewhere
        self.decomposition = None  # must be defined elsewhere
        self.decomposition_intrinsic = None  # must be defined elsewhere
        # Ordered (case, case_mp) list built from PRODUCTS config.
        self.product_stage_sequence = []

        self.coord_buffer = None
        self.to_dissol_pn_buffer = None

        self.fetch_ind = None

        self.aggregated_ind = np.array([[7, 0, 1, 2, 19, 16, 14],
                                        [6, 0, 1, 5, 18, 15, 14],
                                        [8, 0, 4, 5, 20, 15, 17],
                                        [9, 0, 4, 2, 21, 16, 17],
                                        [11, 3, 1, 2, 19, 24, 22],
                                        [10, 3, 1, 5, 18, 23, 22],
                                        [12, 3, 4, 5, 20, 23, 25],
                                        [13, 3, 4, 2, 21, 24, 25]], dtype=np.int64)

        # Intelligent worker allocation to avoid CPU oversubscription
        _cpu_count = os.cpu_count() or 1
        _max_workers = Config.NUMBER_OF_PROCESSES if Config.NUMBER_OF_PROCESSES > 0 else _cpu_count
        
        # Calculate worker allocation
        jmatpro_ratio = Config.JMATPRO_WORKER_RATIO
        ca_workers = max(1, int(_max_workers * (1 - jmatpro_ratio)))
        jmatpro_workers = max(1, _max_workers - ca_workers)
    
        # Ensure we don't exceed CPU count
        ca_workers = min(ca_workers, _cpu_count)
        jmatpro_workers = min(jmatpro_workers, _cpu_count)
        total_workers = ca_workers + jmatpro_workers
        
        # Warn if oversubscription would occur
        if total_workers > _cpu_count:
            print(f"Warning: Total workers ({total_workers}) exceeds CPU count ({_cpu_count}). "
                  f"Consider reducing NUMBER_OF_PROCESSES or setting JMATPRO_WORKER_RATIO.")
            # Cap at CPU count, prioritizing CA workers
            if ca_workers + jmatpro_workers > _cpu_count:
                excess = (ca_workers + jmatpro_workers) - _cpu_count
                jmatpro_workers = max(1, jmatpro_workers - excess)
         
        # Store worker allocation info for debugging
        self.worker_allocation = {
            'ca_workers': ca_workers,
            'jmatpro_workers': jmatpro_workers,
            'total_cpus': _cpu_count,
            'max_configured': _max_workers
        }

        self.comb_indexes = None
        self.rel_prod_fraction = None
        self.gamma_primes = None
        self.product_indexes = None
        self.nucleation_indexes = None
        self.save_flag = False

        self.product_x_not_stab = np.full(self.cells_per_axis, True, dtype=bool)
        # self.TdDATA = td_data.TdDATA()
        # self.TdDATA.fetch_look_up_from_file()
        self.TdDATA = None
        self.jmatpro_pool = None
        # Last non-empty JMatPro phase dict per plane index (aligned with enumerate(task_ids)).
        self._jmatpro_phases_by_plane = {}
        self._jmatpro_phases_by_block = {}
        # self.curr_look_up = None

        # self.TdDATA = JMatProWorkerPool(
        #     temperature=Config.TEMPERATURE,
        #     num_workers=jmatpro_workers,  # Use allocated number, not total
        #     task_timeout=3.0,
        #     max_retries=3
        # )

        # self.KinDATA = kin_data.KinDATA("LUT_NiCr5.pkl")
        # self.KinDATA.fetch_look_up_from_file()
        self.curr_look_up = None

        self.prev_stab_count = 0
        # Tracks concentrations per slab (oxidation-plane index or ignited block id) per product:
        # key=(iteration, product_name, plane_index),
        # value=(jmatpro_conc, existing_conc, diff, cells_conc).
        self.product_plane_tracking = {}

        self._per_iter_log_path = self._resolve_per_iter_log_path()
        if self._per_iter_log_path is not None:
            try:
                os.makedirs(os.path.dirname(self._per_iter_log_path), exist_ok=True)
                with open(self._per_iter_log_path, "w", encoding="utf-8") as f:
                    f.write("# Per-iteration JMatPro state log\n")
                    f.write(f"# matrix={getattr(Config.MATRIX, 'ELEMENT', '?')} ")
                    f.write(f"cells_per_axis={self.cells_per_axis} ")
                    f.write(f"PRECIPITATION_STRIDE={getattr(Config, 'PRECIPITATION_STRIDE', 1)}\n")
            except OSError as exc:
                print(f"[per_iter_log] could not create '{self._per_iter_log_path}': {exc}")
                self._per_iter_log_path = None

        self.precipitation_stride = Config.STRIDE * Config.STRIDE_MULTIPLIER

        # self.save_rate = self.n_iter // Config.STRIDE
        # self.cumul_prod = utils.my_data_structs.MyBufferSingle((self.cells_per_axis, self.save_rate), dtype=float)
        # self.growth_rate = utils.my_data_structs.MyBufferSingle((self.cells_per_axis, self.save_rate), dtype=float)

        self.diffs = None
        self.curr_time = 0

        # lambdas = (np.arange(self.cells_per_axis, dtype=int) + 0.5) * Config.GENERATED_VALUES.LAMBDA
        # adj_lamd = lambdas - Config.ZETTA_ZERO
        # neg_ind = np.where(adj_lamd < 0)[0]
        # adj_lamd[neg_ind] = 0
        # self.active_times = adj_lamd ** 2 / Config.GENERATED_VALUES.KINETIC_KONST ** 2

        # self.prev_len = 0
        # self.powers = utils.physical_data.POWERS

        self.all_phases = []
        self.elem_counts_from_product = {"Ni":0, "O":0, "Cr":0, "Al":0, "N":0}

    def _jmatpro_raw_to_phase_array(self, raw_list):
        """Postprocess raw JMatPro output to (5, n) phase fraction array. raw_list: list of dicts from TdDATA.get_look_up_data.
        Each raw entry can be phase_name -> molar_fraction (float) or phase_name -> dict with 'molar_fraction', 'elements', 'composition'."""
        mapping = getattr(Config, 'JMATPRO_PHASE_MAPPING', None)
        if mapping is None:
            # If the user didn't provide a phase→(5 channels) mapping, default to zeros
            # instead of failing at import time.
            return np.zeros((5, len(raw_list)), dtype=float)
        n = len(raw_list)
        out = np.zeros((5, n), dtype=float)
        for i, raw in enumerate(raw_list):
            if not isinstance(raw, dict):
                continue
            for idx, jmatpro_names in enumerate(mapping):
                for name in jmatpro_names:
                    if name in raw:
                        val = raw[name]
                        out[idx, i] = val["molar_fraction"] if isinstance(val, dict) else val
                        break
        return out

    def ensure_td_lookup(self):
        """Ensure self.TdDATA is the KDTree-based Td lookup object."""
        if self.TdDATA is not None:
            return
        # TdDATA class lives in thermodynamics.td_data and is imported via `from thermodynamics import *`.
        self.TdDATA = TdDATA()
        self.TdDATA.fetch_look_up_from_file()

    def ensure_jmatpro_pool(self):
        """Ensure self.jmatpro_pool is the JMatPro async worker pool from `self.worker_pools`."""
        if self.jmatpro_pool is not None:
            return
        if self.worker_pools is None:
            raise RuntimeError("worker_pools is not set; cannot create JMatPro pool.")

        j_workers = int(self.worker_allocation.get("jmatpro_workers", 1))
        self.jmatpro_pool = self.worker_pools.get_jmatpro_pool(
            num_workers=j_workers,
            temperature=float(Config.TEMPERATURE),
            task_timeout=30.0,
            max_retries=3,
        )

    def _merge_jmatpro_results_with_plane_memory(self, raw_list, task_ids):
        """
        For each plane index i, task_ids[i] maps to one JMatPro result. If that result is an
        empty dict (failure/timeout), substitute the last stored non-empty phases for plane i.
        Successful results refresh the stored copy so memory always tracks the latest good data.
        """
        for plane_idx, tid in enumerate(task_ids):
            phases = raw_list.get(tid)
            if isinstance(phases, dict) and phases:
                self._jmatpro_phases_by_plane[plane_idx] = copy.deepcopy(phases)
            else:
                prev = self._jmatpro_phases_by_plane.get(plane_idx)
                if prev:
                    raw_list[tid] = copy.deepcopy(prev)

    def _merge_jmatpro_results_with_block_memory(self, raw_list, task_ids, block_ids):
        """
        For each ignited block, task_ids[i] maps to one JMatPro result for block_ids[i].
        If that result is an empty dict (failure/timeout), substitute the last stored
        non-empty phases for that block id. Successful results refresh the stored copy.
        """
        for idx, tid in enumerate(task_ids):
            bid = int(block_ids[idx])
            phases = raw_list.get(tid)
            if isinstance(phases, dict) and phases:
                self._jmatpro_phases_by_block[bid] = copy.deepcopy(phases)
            else:
                prev = self._jmatpro_phases_by_block.get(bid)
                if prev:
                    raw_list[tid] = copy.deepcopy(prev)

    def _get_product_counts_upto_bound_for_case(self, case_mp, u_bound):
        ub = int(u_bound)
        pid = int(case_mp.product_phase_id)
        state = self.cases.product_state
        owner = state[0]
        counts = state[1]
        return product_counts_upto_bound_from_state(owner, counts, pid, ub)

    def _get_product_counts_at_indexes_for_case(self, case, case_mp, page_indexes):
        idx = np.asarray(page_indexes, dtype=np.intp).ravel()
        if idx.size == 0:
            return np.zeros(0, dtype=np.uint32)
        pid = int(case_mp.product_phase_id)
        state = self.cases.product_state
        owner = state[0]
        counts = state[1]
        return product_counts_at_indexes_from_state(owner, counts, pid, idx)

    @staticmethod
    def _cfg_elem_map(group):
        out = {}
        for cfg in group:
            if not isinstance(cfg, dict):
                continue
            elem = str(cfg.get("element", "None"))
            if not elem or elem.lower() == "none":
                continue
            out[elem] = SimpleNamespace(
                MOLES_PER_CELL=cfg.get("MOLES_PER_CELL", 0.0),
                MASS_PER_CELL=cfg.get("MASS_PER_CELL", 0.0),
                EQ_MATRIX_MOLES_PER_CELL=cfg.get("EQ_MATRIX_MOLES_PER_CELL", 0.0),
                T=cfg.get("T", 0.0),
            )
        return out

    def _refresh_elem_cfg_cache(self):
        self._oxid_cfg_map = self._cfg_elem_map(Config.OXIDANTS)
        self._act_cfg_map = self._cfg_elem_map(Config.ACTIVES)
        self._act_eq_matrix_map = {
            elem: float(getattr(cfg, "EQ_MATRIX_MOLES_PER_CELL", 0.0))
            for elem, cfg in self._act_cfg_map.items()
        }
        self._act_t_map = {
            elem: float(getattr(cfg, "T", 0.0))
            for elem, cfg in self._act_cfg_map.items()
        }

    def _get_product_cfg_for_case_mp(self, case_mp):
        return getattr(case_mp, "product_cfg", None)

    def _gather_species_counts_by_element(self, u_bound, species_objs):
        out = {}
        for obj in species_objs:
            elem = obj.elem_name
            grid = obj.get_3d_grid()[0]
            counts = np.sum(grid[:u_bound + 1, :, :], axis=(1, 2)).astype(np.float64)
            out[elem] = counts
        return out

    @staticmethod
    def _resolve_per_iter_log_path():
        if not bool(getattr(Config, "LOG_PER_ITER_TEXT", False)):
            return None
        path = str(getattr(Config, "LOG_PER_ITER_PATH", "") or "").strip()
        if path:
            return os.path.abspath(path)
        save_dir = str(getattr(Config, "SAVE_PATH", "") or "").strip() or "."
        return os.path.abspath(os.path.join(save_dir, "per_iter_jmatpro_log.txt"))

    def _log_per_iter_jmatpro_state(
        self,
        oxid_counts,
        act_counts,
        product_counts_by_identifier,
        product_runtime,
        compositions,
        elements,
        product_c_by_identifier,
        jm_target_by_ident,
    ):
        """Append a per-iteration block to the text log.

        The block reports, plane by plane (x = depth):
        - free atoms per element (oxidant/active);
        - atoms bound in products per element (sum over products of cells * stoich);
        - cells per product;
        - composition fed to JMatPro (at%);
        - existing product concentration (p_moles / whole_moles, see line ~705);
        - JMatPro target concentration for the same product identifier.
        """
        if self._per_iter_log_path is None:
            return

        n_planes = int(self.ioz_bound) + 1
        free_elem_names = sorted(set(list(oxid_counts.keys()) + list(act_counts.keys())))
        free_table = np.zeros((n_planes, len(free_elem_names)), dtype=np.float64)
        for j, elem in enumerate(free_elem_names):
            arr = oxid_counts.get(elem, act_counts.get(elem))
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float64)
            n = min(arr.shape[0], n_planes)
            free_table[:n, j] = arr[:n]

        product_idents = [pi for _, pi in product_runtime]
        bound_elem_set = set()
        stoich_per_prod = {}
        for case_mp, ident in product_runtime:
            stoich = dict(getattr(case_mp.product_cfg, "STOICH", {}) or {})
            stoich_per_prod[ident] = {str(k): float(v) for k, v in stoich.items() if float(v) > 0.0}
            bound_elem_set.update(stoich_per_prod[ident].keys())
        bound_elem_names = sorted(bound_elem_set)
        bound_table = np.zeros((n_planes, len(bound_elem_names)), dtype=np.float64)
        for ident in product_idents:
            counts = np.asarray(product_counts_by_identifier.get(ident, np.zeros(n_planes)), dtype=np.float64)
            n = min(counts.shape[0], n_planes)
            for j, elem in enumerate(bound_elem_names):
                nu = stoich_per_prod[ident].get(elem, 0.0)
                if nu > 0.0:
                    bound_table[:n, j] += counts[:n] * nu

        prod_count_table = np.zeros((n_planes, len(product_idents)), dtype=np.float64)
        for j, ident in enumerate(product_idents):
            counts = np.asarray(product_counts_by_identifier.get(ident, np.zeros(n_planes)), dtype=np.float64)
            n = min(counts.shape[0], n_planes)
            prod_count_table[:n, j] = counts[:n]

        comp_arr = np.asarray(compositions, dtype=np.float64)
        if comp_arr.ndim != 2 or comp_arr.shape[0] != n_planes or comp_arr.shape[1] != len(elements):
            comp_arr = np.zeros((n_planes, len(elements)), dtype=np.float64)

        existing_table = np.zeros((n_planes, len(product_idents)), dtype=np.float64)
        for j, ident in enumerate(product_idents):
            arr = np.asarray(product_c_by_identifier.get(ident, np.zeros(n_planes)), dtype=np.float64)
            n = min(arr.shape[0], n_planes)
            existing_table[:n, j] = arr[:n]

        jm_table = np.zeros((n_planes, len(product_idents)), dtype=np.float64)
        for j, ident in enumerate(product_idents):
            arr = np.asarray(jm_target_by_ident.get(ident, np.zeros(n_planes)), dtype=np.float64)
            n = min(arr.shape[0], n_planes)
            jm_table[:n, j] = arr[:n]

        def _fmt_table(headers, table, fmt):
            head = "  plane  " + "  ".join(f"{h:>10s}" for h in headers)
            sep = "  -----  " + "  ".join("-" * 10 for _ in headers)
            lines = [head, sep]
            for k in range(table.shape[0]):
                row = "  ".join(format(table[k, j], fmt) for j in range(table.shape[1]))
                lines.append(f"  {k:>5d}  {row}")
            return "\n".join(lines)

        try:
            with open(self._per_iter_log_path, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write("=" * 90 + "\n")
                f.write(
                    f"ITERATION {int(self.iteration):>8d}    "
                    f"ioz_bound={int(self.ioz_bound):<4d}    "
                    f"matrix={elements[0] if elements else '?'}\n"
                )
                f.write("=" * 90 + "\n")

                if free_elem_names:
                    f.write("\n[ Free atoms per plane (cell counts) ]\n")
                    f.write(_fmt_table(free_elem_names, free_table, ">10.0f") + "\n")

                if bound_elem_names:
                    f.write("\n[ Bound atoms in products per plane (cells x stoich) ]\n")
                    f.write(_fmt_table(bound_elem_names, bound_table, ">10.0f") + "\n")

                if product_idents:
                    f.write("\n[ Product cells per plane ]\n")
                    f.write(_fmt_table(product_idents, prod_count_table, ">10.0f") + "\n")

                if elements:
                    f.write("\n[ Composition fed to JMatPro (at%, matrix first) ]\n")
                    f.write(_fmt_table(elements, comp_arr, ">10.4f") + "\n")

                if product_idents:
                    f.write("\n[ Existing product concentration (p_moles / whole_moles) ]\n")
                    f.write(_fmt_table(product_idents, existing_table, ">10.6f") + "\n")
                    f.write("\n[ JMatPro target product concentration ]\n")
                    f.write(_fmt_table(product_idents, jm_table, ">10.6f") + "\n")
        except OSError as exc:
            print(f"[per_iter_log] write failed: {exc}")
            self._per_iter_log_path = None

    @staticmethod
    def _compute_jmatpro_block_params(n_cells):
        """
        Resolve heterogeneous JMatPro block dimensions from Config.

        Call once at init; ``cells_per_axis`` and block config do not change mid-simulation.

        Returns:
            (cx, cy, cz, Bx, By, Bz)
        where:
            cx,cy,cz = cells per block along x,y,z
            Bx,By,Bz = number of blocks along x,y,z
        """
        cx = int(getattr(Config, "JMATPRO_BLOCK_CELLS_X", 0) or 0)
        cy = int(getattr(Config, "JMATPRO_BLOCK_CELLS_Y", 0) or 0)
        cz = int(getattr(Config, "JMATPRO_BLOCK_CELLS_Z", 0) or 0)
        if cx > 0 and cy > 0 and cz > 0:
            if n_cells % cx != 0 or n_cells % cy != 0 or n_cells % cz != 0:
                raise ValueError(
                    f"N_CELLS_PER_AXIS={n_cells} must be divisible by "
                    f"JMATPRO_BLOCK_CELLS_X/Y/Z={cx}/{cy}/{cz}."
                )
            Bx = n_cells // cx
            By = n_cells // cy
            Bz = n_cells // cz
        else:
            bpa = int(getattr(Config, "JMATPRO_BLOCKS_PER_AXIS", 10) or 10)
            if bpa <= 0:
                bpa = 10
            if n_cells % bpa != 0:
                raise ValueError(
                    f"N_CELLS_PER_AXIS={n_cells} must be divisible by blocks_per_axis={bpa} "
                    "for equal-size 3D subblocks."
                )
            cx = cy = cz = n_cells // bpa
            Bx = By = Bz = bpa
        # For bitmask representation we must keep Bz <= 16.
        if Bz > 16:
            raise ValueError(f"JMatPro block grid has Bz={Bz} > 16; too many z-blocks for uint16 bitmask.")
        return cx, cy, cz, Bx, By, Bz

    def _init_jmatpro_product_scan_cache(self):
        """
        (case_mp, product_cfg) list and phase_id -> row map for product_counts_blocks_ignited_from_state.
        Built once; product_case_pairs and phase ids are fixed for the run.
        """
        product_cases = [
            (case_mp, case_mp.product_cfg) for _, case_mp in self.cases.product_case_pairs
        ]
        pid_to_row = np.full(256, -1, dtype=np.int16)
        row_ix = 0
        for case_mp, _p_cfg in product_cases:
            pid = int(getattr(case_mp, "product_phase_id", 0))
            if pid <= 0:
                continue
            if pid_to_row[pid] >= 0:
                continue
            pid_to_row[pid] = np.int16(row_ix)
            row_ix += 1
        self._jmatpro_product_cases = product_cases
        self._jmatpro_pid_to_row = pid_to_row
        self._jmatpro_n_products = int(row_ix)

    @staticmethod
    def _sum_grid_into_blocks_ignited(grid3d, By, Bz, cx, cy, cz, x_hi):
        """
        Sum a cubic (N,N,N) grid into ignited blocks along x in [0, x_hi).
        Returns float64 vector of length (Bx_ignited * By * Bz).
        Flatten order matches block_id = (bx*By + by)*Bz + bz (bx-major).
        """
        # xh is expected to be a multiple of cx (constructed from bx_max).
        bx_cnt = x_hi // cx
        sub = grid3d[: bx_cnt * cx, :, :]
        resh = sub.reshape(bx_cnt, cx, By, cy, Bz, cz)
        blk = np.sum(resh, axis=(1, 3, 5), dtype=np.uint32)  # (bx_cnt, By, Bz)
        return blk.reshape(-1).astype(np.float64)

    def _gather_species_block_counts_by_element_ignited(self, x_hi, species_objs, By, Bz, cx, cy, cz):
        out = {}
        for obj in species_objs:
            elem = obj.elem_name
            grid = obj.get_3d_grid()[0]
            out[elem] = self._sum_grid_into_blocks_ignited(grid, By, Bz, cx, cy, cz, x_hi)
        return out

    def get_comb_ind_jmatpro_blocks_ignited(self):
        """
        Block-wise composition (sizes from Config) and JMatPro lookup for *ignited* x-blocks only.
        Ignited means bx <= floor(max_inward_x / cx) where cx is the x block thickness in cells.
        Non-ignited blocks are left unchanged.
        """
        self.ensure_jmatpro_pool()
        n = int(self.cells_per_axis)
        cx, cy, cz, Bx, By, Bz = self._jmatpro_block_params
        max_x = int(self.get_cur_ioz_bound())
        bx_max = max_x // cx
        x_hi = min(n, (bx_max + 1) * cx)
        Bx_ignited = x_hi // cx
        n_ignited_blocks = Bx_ignited * (By * Bz)
        block_ids = np.arange(n_ignited_blocks, dtype=np.intp)

        matrix_elem = Config.MATRIX.ELEMENT
        oxid_counts = self._gather_species_block_counts_by_element_ignited(
            x_hi, self.cases.all_oxidants, By, Bz, cx, cy, cz
        )
        act_counts = self._gather_species_block_counts_by_element_ignited(
            x_hi, self.cases.all_actives, By, Bz, cx, cy, cz
        )

        elem_free_moles = {}
        for elem, counts in oxid_counts.items():
            elem_free_moles[elem] = counts * self._oxid_cfg_map[elem].MOLES_PER_CELL
        for elem, counts in act_counts.items():
            elem_free_moles[elem] = counts * self._act_cfg_map[elem].MOLES_PER_CELL

        outward_eq_mat_moles = np.zeros(n_ignited_blocks, dtype=np.float64)
        for elem, counts in act_counts.items():
            outward_eq_mat_moles += counts * self._act_eq_matrix_map[elem]

        product_moles_total = np.zeros(n_ignited_blocks, dtype=np.float64)
        product_eq_mat_moles = np.zeros(n_ignited_blocks, dtype=np.float64)
        product_matrix_pull_moles = np.zeros(n_ignited_blocks, dtype=np.float64)
        elem_pure_moles = dict(elem_free_moles)

        product_cases = self._jmatpro_product_cases
        pid_to_row = self._jmatpro_pid_to_row
        n_products = self._jmatpro_n_products

        prod_counts_rows = product_counts_blocks_ignited_from_state(
            self.cases.product_state[0],
            self.cases.product_state[1],
            pid_to_row,
            n_products,
            Bx,
            By,
            Bz,
            self._jmatpro_n_blocks,
            cx,
            cy,
            cz,
            x_hi,
        )

        product_runtime = []
        product_moles_by_identifier = {}
        for case_idx, (case_mp, p_cfg) in enumerate(product_cases):
            row = pid_to_row[case_mp.product_phase_id]
            # Flattened full block vector, but only the ignited prefix is meaningful.
            p_counts = prod_counts_rows[row, :n_ignited_blocks].astype(np.float64)
            p_moles = p_counts * p_cfg.MOLES_PER_CELL
            product_moles_total += p_moles
            product_moles_by_identifier[p_cfg.ELEMENT] = p_moles

            for elem, frac in case_mp.stoich_frac_items:
                elem_pure_moles[elem] = elem_pure_moles.get(elem, 0.0) + p_moles * frac
            out_elem_case = str(getattr(case_mp, "outward_element", ""))
            thr_out_case = int(getattr(p_cfg, "THRESHOLD_OUTWARD", 0))
            if out_elem_case and thr_out_case > 0:
                product_eq_mat_moles += p_counts * self._act_eq_matrix_map[out_elem_case] * thr_out_case
            product_matrix_pull_moles += p_counts * float(getattr(case_mp, "matrix_moles_per_cell", 0.0))
            product_runtime.append((case_mp, p_cfg.ELEMENT))

        matrix_moles_per_block = float(getattr(Config.MATRIX, "MOLES_PER_CELL", 0.0)) * float(cx * cy * cz)
        matrix_moles = (
            np.full(n_ignited_blocks, matrix_moles_per_block, dtype=np.float64)
            - outward_eq_mat_moles
            - product_eq_mat_moles
            - product_matrix_pull_moles
        )
        whole_moles = matrix_moles + product_moles_total
        for m in elem_free_moles.values():
            whole_moles += m

        product_c_by_identifier = {}
        for p_ident, p_moles in product_moles_by_identifier.items():
            product_c_by_identifier[p_ident] = p_moles / whole_moles

        matrix_moles_pure = np.full(n_ignited_blocks, matrix_moles_per_block, dtype=np.float64)
        for elem, moles in elem_pure_moles.items():
            matrix_moles_pure -= moles * self._act_t_map.get(elem, 0.0)

        comp_elems = [e for e in sorted(elem_pure_moles.keys()) if e and e != matrix_elem]
        elements = [matrix_elem] + comp_elems
        n_comp = len(comp_elems)
        if n_comp > 0:
            comp_stack = np.vstack([elem_pure_moles[e] for e in comp_elems])  # (n_comp, n_blocks)
            non_matrix_sum = np.sum(comp_stack, axis=0)
        else:
            comp_stack = np.zeros((0, n_ignited_blocks), dtype=np.float64)
            non_matrix_sum = np.zeros(n_ignited_blocks, dtype=np.float64)

        tot = matrix_moles_pure + non_matrix_sum
        rows = np.zeros((n_ignited_blocks, n_comp + 1), dtype=np.float64)
        valid = tot > 0.0
        rows[~valid, 0] = 100.0
        if np.any(valid):
            inv_tot = np.zeros(n_ignited_blocks, dtype=np.float64)
            inv_tot[valid] = 100.0 / tot[valid]
            rows[:, 0] = matrix_moles_pure * inv_tot
            for j in range(n_comp):
                rows[:, j + 1] = comp_stack[j] * inv_tot
            row_sum = np.sum(rows, axis=1)
            high = row_sum > 100.0
            if np.any(high):
                rows[high] *= (100.0 / row_sum[high])[:, None]

        compositions = rows.tolist()
        task_ids = self.jmatpro_pool.submit_tasks(compositions, elements=elements)
        raw_list = self.jmatpro_pool.get_results(task_ids, wait=True, timeout=10.0)
        self._merge_jmatpro_results_with_block_memory(raw_list, task_ids, block_ids)

        # Block-wise decision of where to nucleate / severe-collapse.
        bx_cnt = Bx_ignited
        for case_mp, product_ident in product_runtime:
            # new block state
            case_mp.block_mask_bits = np.zeros((bx_cnt, By), dtype=np.uint16)
            case_mp.dissolution_block_mask_bits = np.zeros((bx_cnt, By), dtype=np.uint16)
            case_mp.dissolution_plane_indexes = []
            case_mp.severe_dissolution_block_ids = []
            case_mp.ignited_block_ids = block_ids.tolist()

            out_elem_ref = getattr(case_mp, "outward_element", None)
            jm_identifier = getattr(case_mp, "jm_identifier", None)
            if not jm_identifier:
                continue
            jm_snap_blocks = np.zeros(n_ignited_blocks, dtype=np.float64)
            prod_row = int(pid_to_row[int(case_mp.product_phase_id)])

            for local_idx, tid in enumerate(task_ids):
                bid = int(block_ids[local_idx])
                phases = raw_list.get(tid, {})
                if not phases:
                    continue
                phased = phases.get(jm_identifier)
                if not phased or phased.get("molar_fraction", 0.0) == 0:
                    if int(prod_counts_rows[prod_row, bid]) > 0:
                        case_mp.severe_dissolution_block_ids.append(bid)
                    continue

                product_c_jm = 0.0
                if out_elem_ref:
                    sum_non_ox = float(phased.get("sum_non_ox", 0.0))
                    for jm_elem, jm_comp in zip(phased["elements"], phased["composition"]):
                        if jm_elem == out_elem_ref:
                            product_c_jm = (float(jm_comp) / sum_non_ox) * float(phased.get("molar_fraction", 0.0))
                            break
                else:
                    product_c_jm = float(phased.get("molar_fraction", 0.0))

                jm_snap_blocks[local_idx] = float(product_c_jm)

                existing_c = float(product_c_by_identifier.get(product_ident, np.zeros(1))[bid]) if product_ident in product_c_by_identifier else 0.0
                bx = bid // (By * Bz)
                rem = bid - bx * (By * Bz)
                by = rem // Bz
                bz = rem - by * Bz
                if not (0 <= bx < bx_cnt):
                    continue
                pe = float(Config.PROD_ERROR)
                if product_c_jm > 0.0:
                    rel = (product_c_jm - existing_c) / product_c_jm
                    if rel > pe:
                        case_mp.block_mask_bits[bx, by] = np.uint16(
                            int(case_mp.block_mask_bits[bx, by]) | (1 << int(bz))
                        )
                        case_mp.dissolution_block_mask_bits[bx, by] = np.uint16(
                            int(case_mp.dissolution_block_mask_bits[bx, by]) | (1 << int(bz))
                        )
                    elif rel < -pe:
                        case_mp.dissolution_block_mask_bits[bx, by] = np.uint16(
                            int(case_mp.dissolution_block_mask_bits[bx, by]) | (1 << int(bz))
                        )
                elif existing_c > 0.0:
                    case_mp.dissolution_block_mask_bits[bx, by] = np.uint16(
                        int(case_mp.dissolution_block_mask_bits[bx, by]) | (1 << int(bz))
                    )

            cells_pb = float(int(cx) * int(cy) * int(cz)) or 1.0
            p_c = product_c_by_identifier.get(product_ident)
            for bid in range(n_ignited_blocks):
                jm_b = float(jm_snap_blocks[bid])
                existing_b = float(p_c[bid]) if p_c is not None and bid < p_c.shape[0] else 0.0
                diff_b = jm_b - existing_b
                cells_conc_b = float(prod_counts_rows[prod_row, bid]) / cells_pb
                self.product_plane_tracking[(int(self.iteration), str(product_ident), int(bid))] = (
                    jm_b,
                    existing_b,
                    diff_b,
                    cells_conc_b,
                )

    def _resolve_product_stoich_roles(self, case, case_mp, product_cfg):
        stoich_cfg = getattr(product_cfg, "STOICH", {})
        stoich = {}
        for k, v in stoich_cfg.items():
            try:
                vv = int(v)
            except (TypeError, ValueError):
                continue
            if vv > 0:
                stoich[str(k)] = vv

        outward_elem = str(getattr(product_cfg, "OUTWARD_ELEMENT", ""))
        inward_elem = str(getattr(product_cfg, "INWARD_ELEMENT", ""))
        outward = set([outward_elem]) if outward_elem else set()
        inward = set([inward_elem]) if inward_elem else set()
        if len(outward) == 0 and len(inward) == 0:
            out_elem = str(getattr(getattr(case, "active", None), "elem_name", ""))
            in_elem = str(getattr(getattr(case, "oxidant", None), "elem_name", ""))
            if out_elem:
                outward.add(out_elem)
            if in_elem:
                inward.add(in_elem)
        if len(outward) == 0 and len(inward) > 0:
            outward = set(stoich.keys()) - inward
        if len(inward) == 0 and len(outward) > 0:
            inward = set(stoich.keys()) - outward
        return stoich, outward, inward

    def recalc_elem_counts_from_product(self):
        for _, case_mp in self.cases.product_case_pairs:
            p_cfg = case_mp.product_cfg
            p_counts = self._get_product_counts_upto_bound_for_case(case_mp, 0).astype(np.float64)
           
            for elem, _ in case_mp.stoich_frac_items:
                self.elem_counts_from_product[elem] += int(p_counts[0] * p_cfg.THRESHOLD_INWARD)
            
        for oxidant in self.cases.all_oxidants:
            for elem, counts in self.elem_counts_from_product.items():
                if elem == oxidant.elem_name:
                    oxidant.from_product_counts = counts
        
        self.elem_counts_from_product = {"Ni":0, "O":0, "Cr":0, "Al":0, "N":0}

    def recalc_elem_counts_from_product2(self):
        act_counts = self._gather_species_counts_by_element(0, self.cases.all_actives)
        elem_free_moles = {}
        total_outward_moles = 0.0

        for elem, counts in act_counts.items():
            moles = counts[0] * self._act_cfg_map[elem].MOLES_PER_CELL
            elem_free_moles[elem] = moles
            total_outward_moles += moles


        outward_eq_mat_moles = 0
        for elem, counts in act_counts.items():
            outward_eq_mat_moles += counts[0] * self._act_eq_matrix_map[elem]

        product_moles_total = 0
        product_eq_mat_moles = 0
        product_matrix_pull_moles = 0
        elem_pure_moles = dict(elem_free_moles)
        product_moles_by_identifier = {}
        product_counts_by_identifier = {}
        product_runtime = []
        
        for _, case_mp in self.cases.product_case_pairs:
            p_cfg = case_mp.product_cfg
            p_counts = self._get_product_counts_upto_bound_for_case(case_mp, 0).astype(np.float64)[0]
            product_counts_by_identifier[p_cfg.ELEMENT] = p_counts
            p_moles = p_counts * p_cfg.MOLES_PER_CELL
            product_moles_by_identifier[p_cfg.ELEMENT] = p_moles
            product_moles_total += p_moles

            for elem, frac in case_mp.stoich_frac_items:
                elem_pure_moles[elem] = elem_pure_moles.get(elem, 0.0) + p_moles * frac
            out_elem_case = str(getattr(case_mp, "outward_element", ""))
            thr_out_case = int(getattr(p_cfg, "THRESHOLD_OUTWARD", 0))
            if out_elem_case and thr_out_case > 0:
                product_eq_mat_moles += p_counts * self._act_eq_matrix_map[out_elem_case] * thr_out_case
            product_matrix_pull_moles += p_counts * float(getattr(case_mp, "matrix_moles_per_cell", 0.0))
            product_runtime.append((case_mp, p_cfg.ELEMENT))
        
        b_const = total_outward_moles + product_moles_total + self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles - product_matrix_pull_moles
        for oxidant in self.cases.all_oxidants:
            oxidant.adjusted_cells = int(b_const * oxidant.k_const)

    def get_comb_ind_jmatpro_generic(self):
        self.ioz_bound = self.get_cur_ioz_bound()
        matrix_elem = str(getattr(Config.MATRIX, "ELEMENT", "Ni"))
        oxid_counts = self._gather_species_counts_by_element(self.ioz_bound, self.cases.all_oxidants)
        act_counts = self._gather_species_counts_by_element(self.ioz_bound, self.cases.all_actives)
        elem_free_moles = {}

        for elem, counts in oxid_counts.items():
            elem_free_moles[elem] = counts * self._oxid_cfg_map[elem].MOLES_PER_CELL
        for elem, counts in act_counts.items():
            elem_free_moles[elem] = counts * self._act_cfg_map[elem].MOLES_PER_CELL

        outward_eq_mat_moles = np.zeros(self.ioz_bound + 1, dtype=np.float64)
        for elem, counts in act_counts.items():
            outward_eq_mat_moles += counts * self._act_eq_matrix_map[elem]

        product_moles_total = np.zeros(self.ioz_bound + 1, dtype=np.float64)
        product_eq_mat_moles = np.zeros(self.ioz_bound + 1, dtype=np.float64)
        product_matrix_pull_moles = np.zeros(self.ioz_bound + 1, dtype=np.float64)
        elem_pure_moles = dict(elem_free_moles)
        product_moles_by_identifier = {}
        product_counts_by_identifier = {}
        product_runtime = []
        
        for _, case_mp in self.cases.product_case_pairs:
            p_cfg = case_mp.product_cfg
            p_counts = self._get_product_counts_upto_bound_for_case(case_mp, self.ioz_bound).astype(np.float64)
            product_counts_by_identifier[p_cfg.ELEMENT] = p_counts
            p_moles = p_counts * p_cfg.MOLES_PER_CELL
            product_moles_by_identifier[p_cfg.ELEMENT] = p_moles
            product_moles_total += p_moles

            for elem, frac in case_mp.stoich_frac_items:
                elem_pure_moles[elem] = elem_pure_moles.get(elem, 0.0) + p_moles * frac
            out_elem_case = str(getattr(case_mp, "outward_element", ""))
            thr_out_case = int(getattr(p_cfg, "THRESHOLD_OUTWARD", 0))
            if out_elem_case and thr_out_case > 0:
                product_eq_mat_moles += p_counts * self._act_eq_matrix_map[out_elem_case] * thr_out_case
            product_matrix_pull_moles += p_counts * float(getattr(case_mp, "matrix_moles_per_cell", 0.0))
            product_runtime.append((case_mp, p_cfg.ELEMENT))

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles - product_matrix_pull_moles
        whole_moles = matrix_moles + product_moles_total
        for m in elem_free_moles.values():
            whole_moles += m

        product_c_by_identifier = {}
        for p_ident, p_moles in product_moles_by_identifier.items():
            product_c_by_identifier[p_ident] = p_moles / whole_moles

        matrix_moles_pure = np.full(self.ioz_bound + 1, self.matrix_moles_per_page, dtype=np.float64)
        for elem, moles in elem_pure_moles.items():
            matrix_moles_pure -= moles * self._act_t_map.get(elem, 0.0)

        comp_elems = [e for e in sorted(elem_pure_moles.keys()) if e and e != matrix_elem]
        elements = [matrix_elem] + comp_elems
        n_planes = self.ioz_bound + 1
        n_comp = len(comp_elems)
        if n_comp > 0:
            comp_stack = np.vstack([elem_pure_moles[e] for e in comp_elems])  # (n_comp, n_planes)
            non_matrix_sum = np.sum(comp_stack, axis=0)
        else:
            comp_stack = np.zeros((0, n_planes), dtype=np.float64)
            non_matrix_sum = np.zeros(n_planes, dtype=np.float64)
        tot = matrix_moles_pure + non_matrix_sum
        rows = np.zeros((n_planes, n_comp + 1), dtype=np.float64)
        valid = tot > 0.0
        rows[~valid, 0] = 100.0
        if np.any(valid):
            inv_tot = np.zeros(n_planes, dtype=np.float64)
            inv_tot[valid] = 100.0 / tot[valid]
            rows[:, 0] = matrix_moles_pure * inv_tot
            for j in range(n_comp):
                rows[:, j + 1] = comp_stack[j] * inv_tot
            row_sum = np.sum(rows, axis=1)
            high = row_sum > 100.0
            if np.any(high):
                rows[high] *= (100.0 / row_sum[high])[:, None]
        compositions = rows.tolist()

        task_ids = self.jmatpro_pool.submit_tasks(compositions, elements=elements)
        raw_list = self.jmatpro_pool.get_results(task_ids, wait=True, timeout=10.0)
        self._merge_jmatpro_results_with_plane_memory(raw_list, task_ids)

        n_planes_log = self.ioz_bound + 1
        jm_target_by_ident = {ident: np.zeros(n_planes_log, dtype=np.float64) for _, ident in product_runtime}

        for case_mp, product_ident in product_runtime:
            case_mp.plane_indexes = []
            case_mp.dissolution_plane_indexes = []
            case_mp.severe_dissolution_indexes = []
            out_elem_ref = case_mp.outward_element
            jm_identifier = case_mp.jm_identifier
            for plane_idx, tid in enumerate(task_ids):
                phases = raw_list.get(tid, {})
                if not phases:
                    continue
                phased = phases.get(jm_identifier)
                if not phased or phased.get("molar_fraction", 0.0) == 0:
                    plane_counts = product_counts_by_identifier.get(product_ident, None)
                    if int(plane_counts[plane_idx]) > 0:
                        case_mp.severe_dissolution_indexes.append(plane_idx)
                    continue
                product_c_jm = 0.0
                if out_elem_ref:
                    sum_non_ox = float(phased.get("sum_non_ox", 0.0))
                    if sum_non_ox > 0.0:
                        for jm_elem, jm_comp in zip(phased["elements"], phased["composition"]):
                            if jm_elem == out_elem_ref:
                                product_c_jm = (float(jm_comp) / sum_non_ox) * float(phased.get("molar_fraction", 0.0))
                                break
                else:
                    # No outward reactant: use phase fraction directly.
                    product_c_jm = float(phased.get("molar_fraction", 0.0))
                jm_target_by_ident[product_ident][plane_idx] = float(product_c_jm)
                existing_c = float(product_c_by_identifier[product_ident][plane_idx])
                
                if (product_c_jm - existing_c)/product_c_jm > Config.PROD_ERROR and oxid_counts[case_mp.product_cfg.INWARD_ELEMENT][plane_idx] > 10000:
                    case_mp.plane_indexes.append(plane_idx)
                    case_mp.dissolution_plane_indexes.append(plane_idx)
                elif (product_c_jm - existing_c)/product_c_jm < -Config.PROD_ERROR:
                    case_mp.dissolution_plane_indexes.append(plane_idx)
               
                # elif case_mp.dissolution_counter[plane_idx] < case_mp.dissolution_n_iterations:
                #     case_mp.plane_indexes.append(plane_idx)
                #     case_mp.dissolution_plane_indexes.append(plane_idx)
                #     case_mp.dissolution_counter[plane_idx] += 1

            if len(case_mp.plane_indexes) > 1:
                case_mp.plane_indexes = sorted(set(case_mp.plane_indexes))
            if len(case_mp.dissolution_plane_indexes) > 1:
                case_mp.dissolution_plane_indexes = sorted(set(case_mp.dissolution_plane_indexes))
            cells_per_plane = float(self.cells_per_page) if self.cells_per_page else 1.0
            for plane_idx in range(n_planes_log):
                jm_i = float(jm_target_by_ident[product_ident][plane_idx])
                existing_i = float(product_c_by_identifier[product_ident][plane_idx])
                diff_i = jm_i - existing_i
                cells_conc_i = float(product_counts_by_identifier[product_ident][plane_idx]) / cells_per_plane
                self.product_plane_tracking[(int(self.iteration), str(product_ident), int(plane_idx))] = (
                    jm_i,
                    existing_i,
                    diff_i,
                    cells_conc_i,
                )

        if self._per_iter_log_path is not None:
            self._log_per_iter_jmatpro_state(
                oxid_counts=oxid_counts,
                act_counts=act_counts,
                product_counts_by_identifier=product_counts_by_identifier,
                product_runtime=product_runtime,
                compositions=compositions,
                elements=elements,
                product_c_by_identifier=product_c_by_identifier,
                jm_target_by_ident=jm_target_by_ident,
            )

    def diffuse_all(self):
        if Config.RECALC_ELEM_COUNTS_FROM_PRODUCT:
            self.recalc_elem_counts_from_product2()

        run_outward = (self.iteration + 1) % Config.STRIDE == 0
        for e in self.cases.all_actives:
            e.skip_diffusion_this_step = not run_outward
        elems_to_diffuse = list(self.cases.all_oxidants) + list(self.cases.all_actives)
        self.diffusion_engine.diffuse_multiple(elems_to_diffuse)

        for elem in self.cases.all_oxidants:
            elem.fill_first_page()
        if run_outward:
            for elem in self.cases.all_actives:
                elem.fill_last_page()

    def fix_init_precip_bool(self, u_bound, l_bound=0):
        if u_bound == self.cells_per_axis - 1:
            u_bound = self.cells_per_axis - 2
        if l_bound - 1 < 0:
            l_bound = 1
        self.cur_case.precip_3d_init[:, :, l_bound-1:u_bound + 2] = False
        self.cur_case.precip_3d_init[:, :, l_bound-1:u_bound + 2] = self.cur_case.product.c3d[:, :, l_bound-1:u_bound + 2]

    def fix_init_precip_int(self, u_bound):
        if u_bound == self.cells_per_axis - 1:
            u_bound = self.cells_per_axis - 2
        x_hi = u_bound + 2
        self.cases.precip_3d_init[0:x_hi, :, :] = 0
        self.cases.precip_3d_init[0:x_hi, :, :] = self.cases.product_state[1, 0:x_hi, :, :]

    def fix_init_precip_dummy(self, u_bound, l_bound=0):
        pass

    def get_active_oxidant_mutual_indexes(self, oxidant, active):
        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]
        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            index = np.where(oxidant_indexes >= min_act - 1)[0]
            return oxidant_indexes[index]
        else:
            return [self.furthest_index]

    @staticmethod
    def generate_batch_fetch_ind_mp(ranges, size, switch=False):
        iter_shifts = np.array(np.where(np.ones((size, size)) == 1)).transpose()
        dummy_grid = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS), False)
        if switch:
            dummy_grid[ranges[0][0]:ranges[0][1], :] = True
            dummy_grid[ranges[1][0]:ranges[1][1], :] = True
        else:
            dummy_grid[ranges[0]:ranges[1], :] = True
        n_fetch_batch = []
        all_coord = np.array(np.nonzero(dummy_grid), dtype=np.short)
        for step, t in enumerate(iter_shifts):
            t_ind = np.where(((all_coord[0] - t[1]) % size == 0) & ((all_coord[1] - t[0]) % size == 0))[0]
            if len(t_ind) > 0:
                n_fetch_batch.append(all_coord[:, t_ind])
        return n_fetch_batch

    def _map_dissolution_pool(self, worker_fn, tasks):
        """
        Run dissolution workers via mp.Pool.map.

        On some Windows setups, Pool can report zero workers and map() raises ZeroDivisionError.
        Fall back to sequential in-process execution so dissolution still completes.
        """
        if not tasks:
            return
        pool = getattr(self.worker_pools, "dissolution_pool", None)
        if pool is None:
            for t in tasks:
                worker_fn(t)
            return
        try:
            internal = getattr(pool, "_pool", None)
            if internal is not None and len(internal) == 0:
                for t in tasks:
                    worker_fn(t)
                return
        except Exception:
            pass
        try:
            pool.map(worker_fn, tasks)
        except ZeroDivisionError:
            for t in tasks:
                worker_fn(t)

    def dissolution_mp_subblock(self):
        """
        Dissolution V2: product snapshot for consistent neighbour reads; workers write directly
        to oxidant write buffer (count + dirs grid) and active. No flat cells/dirs; oxidant is
        in shared memory. Optional block logic via aggregated_ind and bsf.

        When ``dissolution_block_mask_bits`` is present and has any bit set, uses per-block
        ignited-x dissolution (new worker/kernels); otherwise uses ``dissolution_plane_indexes``.
        """
        dp = getattr(self.cur_case_mp, "dissolution_probabilities", None)
        if dp is None:
            dp = getattr(self.cur_case, "dissolution_probabilities", None)
        if dp is None:
            return
        # Diffusion buffers ping-pong between A/B every step. Dissolution workers must read/write
        # the *current* read buffer; otherwise released particles land in the stale buffer and
        # are lost on the next swap.
        if self.cur_case is not None and self.cur_case_mp is not None:
            if getattr(self.cur_case, "oxidant", None) is not None and hasattr(self.cur_case.oxidant, "get_current_c3d_shm_mdata"):
                self.cur_case_mp.oxidant_c3d_shm_mdata = self.cur_case.oxidant.get_current_c3d_shm_mdata()
            if getattr(self.cur_case, "active", None) is not None and hasattr(self.cur_case.active, "get_current_c3d_shm_mdata"):
                self.cur_case_mp.active_c3d_shm_mdata = self.cur_case.active.get_current_c3d_shm_mdata()
        dbm = getattr(self.cur_case_mp, "dissolution_block_mask_bits", None)
        if dbm is not None and isinstance(dbm, np.ndarray) and dbm.size > 0 and np.any(dbm):
            self._dissolution_mp_subblock_blockmask(dp)
            return
        comb_src = getattr(self.cur_case_mp, "dissolution_plane_indexes", None)
        if comb_src is None or len(comb_src) == 0:
            comb_src = self.comb_indexes
        if comb_src is None:
            return
        # Lists may contain None placeholders; np.asarray(..., dtype=np.intp) fails on None.
        comb_flat = []
        for x in np.asarray(comb_src, dtype=object).ravel():
            if x is None:
                continue
            try:
                comb_flat.append(int(x))
            except (TypeError, ValueError):
                continue
        comb = np.asarray(comb_flat, dtype=np.intp).ravel()
        if comb.size == 0:
            return
        oxidant_elem = self.cur_case.oxidant
        if not hasattr(oxidant_elem, "get_current_c3d_shm_mdata"):
            return
        n_workers = max(1, getattr(self.worker_pools, "n_outward_workers", getattr(self.diffusion_engine, "n_outward_workers", 4)))
        if getattr(self.cur_case_mp, "oxidant_c3d_shm_mdata", None) is None:
            return
        n_z = self.cur_case_mp.oxidant_c3d_shm_mdata.shape[2]
        # Partition z into contiguous slabs (no gaps), like diffusion
        base = n_z // n_workers
        extra = n_z % n_workers
        z_ranges = []
        k = 0
        for w in range(n_workers):
            size = base + (1 if w < extra else 0)
            if size <= 0:
                break
            k_hi = min(k + size - 1, n_z - 1)
            z_ranges.append((k, k_hi))
            k = k_hi + 1
        if not z_ranges:
            return
        block_patterns = get_block_patterns_from_aggregated(getattr(self, "aggregated_ind", None))
        bsf = float(getattr(dp, "bsf", 1.0))
        if bsf < 1.0:
            bsf = 1.0

        product_snapshot_mdata = None

        # Add new oxidant to current (read) buffer so particles are in the active state and survive next diffusion
        oxidant_write_mdata = oxidant_elem.get_current_c3d_shm_mdata()
        max_per_cell_oxidant = oxidant_elem.max_per_cell
        active_elem = self.cur_case.active
        max_per_cell_active = active_elem.max_per_cell if active_elem is not None else 0
        packed_dirs = np.asarray(_DIRS_6_PACKED, dtype=np.uint8)

        values_pp = np.asarray(dp.dissol_prob.values_pp, dtype=np.float64)
        const_a_pp = np.asarray(dp.const_a_pp, dtype=np.float64)
        const_b_pp = np.asarray(dp.const_b_pp, dtype=np.float64)
        const_c_pp = np.asarray(dp.const_c_pp, dtype=np.float64)
        const_d_pp = np.asarray(dp.const_d_pp, dtype=np.float64)

        tasks = [
            (
                self.cur_case_mp,
                comb,
                k_lo,
                k_hi,
                values_pp,
                const_a_pp,
                const_b_pp,
                const_c_pp,
                const_d_pp,
                product_snapshot_mdata,
                oxidant_write_mdata,
                max_per_cell_oxidant,
                max_per_cell_active,
                packed_dirs,
                block_patterns,
                bsf,
            )
            for (k_lo, k_hi) in z_ranges
        ]

        self._map_dissolution_pool(dissolution_subblock_worker, tasks)

    def _dissolution_mp_subblock_blockmask(self, dp):
        """Dissolution limited to (bx,by,bz) cells marked in ``dissolution_block_mask_bits``."""
        oxidant_elem = self.cur_case.oxidant
        if not hasattr(oxidant_elem, "get_current_c3d_shm_mdata"):
            return
        if getattr(self.cur_case_mp, "oxidant_c3d_shm_mdata", None) is None:
            return
        n_workers = max(
            1,
            getattr(
                self.worker_pools,
                "n_outward_workers",
                getattr(self.diffusion_engine, "n_outward_workers", 4),
            ),
        )
        n_z = self.cur_case_mp.oxidant_c3d_shm_mdata.shape[2]
        base = n_z // n_workers
        extra = n_z % n_workers
        z_ranges = []
        k = 0
        for w in range(n_workers):
            size = base + (1 if w < extra else 0)
            if size <= 0:
                break
            k_hi = min(k + size - 1, n_z - 1)
            z_ranges.append((k, k_hi))
            k = k_hi + 1
        if not z_ranges:
            return
        block_patterns = get_block_patterns_from_aggregated(getattr(self, "aggregated_ind", None))
        bsf = float(getattr(dp, "bsf", 1.0))
        if bsf < 1.0:
            bsf = 1.0

        oxidant_write_mdata = oxidant_elem.get_current_c3d_shm_mdata()
        max_per_cell_oxidant = oxidant_elem.max_per_cell
        active_elem = self.cur_case.active
        max_per_cell_active = active_elem.max_per_cell if active_elem is not None else 0
        packed_dirs = np.asarray(_DIRS_6_PACKED, dtype=np.uint8)

        values_pp = np.asarray(dp.dissol_prob.values_pp, dtype=np.float64)
        const_a_pp = np.asarray(dp.const_a_pp, dtype=np.float64)
        const_b_pp = np.asarray(dp.const_b_pp, dtype=np.float64)
        const_c_pp = np.asarray(dp.const_c_pp, dtype=np.float64)
        const_d_pp = np.asarray(dp.const_d_pp, dtype=np.float64)

        cx, cy, cz, _Bx, _By, _Bz = self._jmatpro_block_params
        dbm = self.cur_case_mp.dissolution_block_mask_bits
        x_hi = int(dbm.shape[0]) * int(cx)
        plane_indexes = np.arange(0, min(int(self.cells_per_axis), x_hi), dtype=np.intp)

        tasks = [
            (
                self.cur_case_mp,
                k_lo,
                k_hi,
                plane_indexes,
                values_pp,
                const_a_pp,
                const_b_pp,
                const_c_pp,
                const_d_pp,
                oxidant_write_mdata,
                max_per_cell_oxidant,
                max_per_cell_active,
                packed_dirs,
                block_patterns,
                bsf,
                dbm,
                int(cx),
                int(cy),
                int(cz),
            )
            for (k_lo, k_hi) in z_ranges
        ]
        self._map_dissolution_pool(dissolution_subblock_worker_blockmask, tasks)

    def apply_severe_plane_collapse(self, case, case_mp):
        """
        JMatPro no longer reports the product phase on these x-planes: strip that product from
        the whole (y,z) slice and release inward/outward particles using the same per-unit
        thresholds as dissolution (threshold_inward / threshold_outward per state_count unit).
        Writes current oxidant/active shared segments (same as dissolution V2 write target).
        """
        
        plane_indexes = np.asarray(case_mp.severe_dissolution_indexes, dtype=np.intp).ravel()
        oxidant_elem = case.oxidant
        n_i, n_j, n_z = case_mp.oxidant_c3d_shm_mdata.shape
        max_ox = int(oxidant_elem.max_per_cell)
        shm_ox = shared_memory.SharedMemory(name=oxidant_elem.get_current_c3d_shm_mdata().name)
        oxidant_count, oxidant_dirs = _views_from_segment_dissol(shm_ox, n_i, max_ox)

        active_elem = case.active
        if active_elem is not None and getattr(case_mp, "active_c3d_shm_mdata", None) is not None:
            max_act = int(active_elem.max_per_cell)
            shm_a = shared_memory.SharedMemory(name=active_elem.get_current_c3d_shm_mdata().name)
            active_count, active_dirs = _views_from_segment_dissol(shm_a, n_i, max_act)
        else:
            max_act = 0
            active_count = np.zeros((n_i * n_j * n_z,), dtype=np.uint16)
            active_dirs = np.zeros((n_i * n_j * n_z, 1), dtype=np.uint8)

        shm_state = shared_memory.SharedMemory(name=case_mp.product_state_shm_mdata.name)
        product_state = np.ndarray(
            case_mp.product_state_shm_mdata.shape,
            dtype=case_mp.product_state_shm_mdata.dtype,
            buffer=shm_state.buf,
        )
        owner_phase = product_state[0]
        state_count = product_state[1]
        thr_in = int(getattr(case_mp, "threshold_inward", 1))
        thr_out = int(getattr(case_mp, "threshold_outward", 0))
        dissolution_thresholds = np.array([thr_in, thr_out], dtype=np.int32)
        packed_dirs = np.asarray(_DIRS_6_PACKED, dtype=np.uint8).ravel()
        seed = int(np.random.randint(0, 2**31))
        pid = int(getattr(case_mp, "product_phase_id", 0))

        severe_planes_clear_product_release(
            owner_phase,
            state_count,
            pid,
            plane_indexes,
            n_i,
            n_j,
            n_z,
            oxidant_count,
            oxidant_dirs,
            active_count,
            active_dirs,
            dissolution_thresholds,
            max_ox,
            max_act,
            packed_dirs,
            seed,
        )

        shm_ox.close()
        if active_elem is not None and getattr(case_mp, "active_c3d_shm_mdata", None) is not None:
            shm_a.close()
        shm_state.close()
        case_mp.severe_dissolution_indexes = []

    def apply_severe_block_collapse(self, case, case_mp):
        """
        Block analogue of severe plane collapse: clear product only inside severe blocks and
        release inward/outward particles per state_count unit.
        """
        block_ids = np.asarray(case_mp.severe_dissolution_block_ids, dtype=np.intp).ravel()

        oxidant_elem = case.oxidant
        n_i, n_j, n_z = case_mp.oxidant_c3d_shm_mdata.shape
        max_ox = int(oxidant_elem.max_per_cell)
        shm_ox = shared_memory.SharedMemory(name=oxidant_elem.get_current_c3d_shm_mdata().name)
        oxidant_count, oxidant_dirs = _views_from_segment_dissol(shm_ox, n_i, max_ox)

        active_elem = case.active
        if active_elem is not None and getattr(case_mp, "active_c3d_shm_mdata", None) is not None:
            max_act = int(active_elem.max_per_cell)
            shm_a = shared_memory.SharedMemory(name=active_elem.get_current_c3d_shm_mdata().name)
            active_count, active_dirs = _views_from_segment_dissol(shm_a, n_i, max_act)
        else:
            max_act = 0
            active_count = np.zeros((n_i * n_j * n_z,), dtype=np.uint16)
            active_dirs = np.zeros((n_i * n_j * n_z, 1), dtype=np.uint8)

        shm_state = shared_memory.SharedMemory(name=case_mp.product_state_shm_mdata.name)
        product_state = np.ndarray(
            case_mp.product_state_shm_mdata.shape,
            dtype=case_mp.product_state_shm_mdata.dtype,
            buffer=shm_state.buf,
        )
        owner_phase = product_state[0]
        state_count = product_state[1]
        thr_in = int(getattr(case_mp, "threshold_inward", 1))
        thr_out = int(getattr(case_mp, "threshold_outward", 0))
        dissolution_thresholds = np.array([thr_in, thr_out], dtype=np.int32)
        packed_dirs = np.asarray(_DIRS_6_PACKED, dtype=np.uint8).ravel()
        seed = int(np.random.randint(0, 2**31))
        pid = int(getattr(case_mp, "product_phase_id", 0))

        cx, cy, cz, Bx, By, Bz = self._jmatpro_block_params
        severe_blocks_clear_product_release(
            owner_phase,
            state_count,
            pid,
            block_ids,
            n_i,
            n_j,
            n_z,
            oxidant_count,
            oxidant_dirs,
            active_count,
            active_dirs,
            dissolution_thresholds,
            max_ox,
            max_act,
            packed_dirs,
            seed,
            By,
            Bz,
            cx,
            cy,
            cz,
        )

        shm_ox.close()
        if active_elem is not None and getattr(case_mp, "active_c3d_shm_mdata", None) is not None:
            shm_a.close()
        shm_state.close()
        case_mp.severe_dissolution_block_ids = []

    def _build_precip_mirror_state(self, n_cells, z_ranges_base):
        """Build mirrored periodic z partition (blocks + flattened ranges + gap groups)."""
        def _periodic_groups_from_mask(mask):
            groups = []
            start = None
            for kk in range(n_cells):
                if mask[kk] and start is None:
                    start = kk
                elif not mask[kk] and start is not None:
                    groups.append([start, kk - 1])
                    start = None
            if start is not None:
                groups.append([start, n_cells - 1])
            if len(groups) >= 2 and groups[0][0] == 0 and groups[-1][1] == n_cells - 1:
                groups = [[groups[-1][0], groups[0][1]]] + groups[1:-1]
            return groups

        block_mask = np.zeros(n_cells, dtype=np.bool_)
        for k_lo, k_hi in z_ranges_base:
            block_mask[int(k_lo):int(k_hi) + 1] = True
        gap_mask = ~block_mask

        base_block_groups = _periodic_groups_from_mask(block_mask)
        base_gap_groups = _periodic_groups_from_mask(gap_mask)
        block_centers = [0.5 * (g[0] + g[1]) for g in base_block_groups]
        gap_centers = [0.5 * (g[0] + g[1]) for g in base_gap_groups]

        # Pick integer periodic shift that aligns block centers to old gap centers.
        best_shift = 0
        best_score = -1
        target_half = n_cells / 2.0
        for s in range(1, n_cells):
            shifted = [((c + s) % n_cells) for c in block_centers]
            score = sum(any(abs(sc - gc) < 1e-12 for gc in gap_centers) for sc in shifted)
            if score > best_score or (score == best_score and abs(s - target_half) < abs(best_shift - target_half)):
                best_score = score
                best_shift = s

        # Shift each base block with full periodic wrapping.
        z_blocks_neg = []
        z_ranges_neg = []
        block_mask_neg = np.zeros(n_cells, dtype=np.bool_)
        for (k_lo, k_hi) in z_ranges_base:
            mapped = [((int(kk) + best_shift) % n_cells) for kk in range(int(k_lo), int(k_hi) + 1)]
            mapped_sorted = sorted(mapped)
            segs = []
            seg_start = mapped_sorted[0]
            seg_prev = mapped_sorted[0]
            for kk in mapped_sorted[1:]:
                if kk == seg_prev + 1:
                    seg_prev = kk
                else:
                    segs.append((int(seg_start), int(seg_prev)))
                    seg_start = kk
                    seg_prev = kk
            segs.append((int(seg_start), int(seg_prev)))
            if len(segs) == 2 and segs[0][0] == 0 and segs[1][1] == n_cells - 1:
                segs = [segs[1], segs[0]]
            z_blocks_neg.append(segs)
            for a, b in segs:
                z_ranges_neg.append((int(a), int(b)))
                block_mask_neg[int(a):int(b) + 1] = True

        gap_z_set_neg = {k for k in range(n_cells) if not block_mask_neg[k]}
        gap_z_groups_neg = _partition_gap_z_parallel(gap_z_set_neg, min_spacing=3, n_z=n_cells)
        return z_blocks_neg, z_ranges_neg, gap_z_groups_neg

    def _ensure_precip_z_states(self):
        """Initialize (or rebuild) base and mirrored z states once per (n_cells, n_workers)."""
        part_sig = (int(self.cells_per_axis), int(self.worker_pools.n_inward_workers))
        if self._precip_partition_sig == part_sig and self._precip_z_blocks is not None and self._precip_z_blocks_neg is not None:
            return

        z_ranges, gap_z_set = _partition_domain_z(self.cells_per_axis, self.worker_pools.n_inward_workers)
        gap_z_groups = _partition_gap_z_parallel(gap_z_set, min_spacing=3, n_z=self.cells_per_axis)
        z_blocks = [[(int(k_lo), int(k_hi))] for (k_lo, k_hi) in z_ranges]

        z_blocks_neg, z_ranges_neg, gap_z_groups_neg = self._build_precip_mirror_state(self.cells_per_axis, z_ranges)

        self._precip_partition_sig = part_sig
        self._precip_z_ranges = z_ranges
        self._precip_gap_z_groups = gap_z_groups
        self._precip_z_blocks = z_blocks
        self._precip_z_blocks_neg = z_blocks_neg
        self._precip_z_ranges_neg = z_ranges_neg
        self._precip_gap_z_groups_neg = gap_z_groups_neg

    def precip_mp_subblock(self, cur_case, cur_case_mp):
        # IMPORTANT: oxidant/active diffusion buffers are ping-ponged (swap each diffusion step).
        # Nucleation/precipitation workers must attach to the *current* diffusion read buffer,
        # otherwise they will consume/react in the wrong SHM segment every other iteration.
        cur_case_mp.oxidant_c3d_shm_mdata = cur_case.oxidant.get_current_c3d_shm_mdata()
        if cur_case.active is not None:
            cur_case_mp.active_c3d_shm_mdata = cur_case.active.get_current_c3d_shm_mdata()

        cur_case.fix_init_precip_func_ref(self.ioz_bound)

        # Two-state strategy only: even iterations -> base, odd iterations -> mirrored.
        if int(self.iteration) % 2 == 0:
            z_blocks = self._precip_z_blocks
            gap_z_groups = self._precip_gap_z_groups
        else:
            z_blocks = self._precip_z_blocks_neg
            gap_z_groups = self._precip_gap_z_groups_neg

        # Legacy path uses explicit plane indexes; block path uses contiguous x-range + block mask.
        block_mask_bits = getattr(cur_case_mp, "block_mask_bits", None)
        if block_mask_bits is not None and isinstance(block_mask_bits, np.ndarray) and block_mask_bits.size > 0:
            cx, cy, cz, Bx, By, Bz = self._jmatpro_block_params
            x_hi = int(block_mask_bits.shape[0]) * int(cx)
            plane_indexes = np.arange(0, min(int(self.cells_per_axis), x_hi), dtype=np.intp)
        else:
            block_mask_bits = None
            cx, cy, cz = 0, 0, 0
            plane_indexes = np.asarray(cur_case_mp.plane_indexes, dtype=np.intp)
        ind_form = np.asarray(ind_formation, dtype=np.int8)

        # Interior z-blocks:
        tasks_std = []
        tasks_tail_seq = []
        for segs in z_blocks:
            if len(segs) == 1:
                k_lo, k_hi = segs[0]
                max_per_cell_active = cur_case.active.max_per_cell if cur_case.active is not None else 0
                if block_mask_bits is None:
                    tasks_std.append((cur_case_mp, k_lo, k_hi, plane_indexes, cur_case.oxidant.max_per_cell, max_per_cell_active, ind_form))
                else:
                    tasks_std.append(
                        (
                            cur_case_mp,
                            k_lo,
                            k_hi,
                            plane_indexes,
                            cur_case.oxidant.max_per_cell,
                            max_per_cell_active,
                            ind_form,
                            block_mask_bits,
                            cx,
                            cy,
                            cz,
                        )
                    )
            else:
                # Use the largest segment in the parallel pass; defer the remaining
                # wrapped segment(s) to a short sequential tail pass.
                segs_sorted = sorted(segs, key=lambda ab: (ab[1] - ab[0] + 1), reverse=True)
                main_seg = segs_sorted[0]
                tail_segs = segs_sorted[1:]
                k_lo, k_hi = main_seg
                max_per_cell_active = cur_case.active.max_per_cell if cur_case.active is not None else 0
                if block_mask_bits is None:
                    tasks_std.append((cur_case_mp, k_lo, k_hi, plane_indexes, cur_case.oxidant.max_per_cell, max_per_cell_active, ind_form))
                else:
                    tasks_std.append(
                        (
                            cur_case_mp,
                            k_lo,
                            k_hi,
                            plane_indexes,
                            cur_case.oxidant.max_per_cell,
                            max_per_cell_active,
                            ind_form,
                            block_mask_bits,
                            cx,
                            cy,
                            cz,
                        )
                    )
                for k_lo, k_hi in tail_segs:
                    if block_mask_bits is None:
                        tasks_tail_seq.append((cur_case_mp, k_lo, k_hi, plane_indexes, cur_case.oxidant.max_per_cell, max_per_cell_active, ind_form))
                    else:
                        tasks_tail_seq.append(
                            (
                                cur_case_mp,
                                k_lo,
                                k_hi,
                                plane_indexes,
                                cur_case.oxidant.max_per_cell,
                                max_per_cell_active,
                                ind_form,
                                block_mask_bits,
                                cx,
                                cy,
                                cz,
                            )
                        )
        if tasks_std:
            self.worker_pools.nucleation_pool.map(precip_step_subblock_worker, tasks_std)
        for task in tasks_tail_seq:
            precip_step_subblock_worker(task)
        # Gap z-planes: run all gap indices in one worker task using explicit seed list.
        gap_seed_slab_k = list(dict.fromkeys(int(k) for group in gap_z_groups for k in group))
        if block_mask_bits is None:
            gap_task = [(
                cur_case_mp,
                0,
                int(cur_case_mp.oxidant_c3d_shm_mdata.shape[2]) - 1,
                plane_indexes,
                cur_case.oxidant.max_per_cell,
                (cur_case.active.max_per_cell if cur_case.active is not None else 0),
                ind_form,
                gap_seed_slab_k,
            )]
        else:
            gap_task = [
                (
                    cur_case_mp,
                    0,
                    int(cur_case_mp.oxidant_c3d_shm_mdata.shape[2]) - 1,
                    plane_indexes,
                    cur_case.oxidant.max_per_cell,
                    (cur_case.active.max_per_cell if cur_case.active is not None else 0),
                    ind_form,
                    block_mask_bits,
                    cx,
                    cy,
                    cz,
                    gap_seed_slab_k,
                )
            ]
        self.worker_pools.nucleation_pool.map(precip_step_subblock_worker, gap_task)
    
    def nucleate(self):
        if self.iteration % Config.PRECIPITATION_STRIDE == 0:
            self.get_combi_ind()

            # Apply severe plane collapse first to ensure correct product release
            for case, case_mp in self.cases.product_case_pairs:
                if case_mp.severe_dissolution_indexes:
                    self.apply_severe_plane_collapse(case, case_mp)
                if getattr(case_mp, "severe_dissolution_block_ids", None):
                    self.apply_severe_block_collapse(case, case_mp)

            for case, case_mp in self.cases.product_case_pairs:
                self.cur_case = case
                self.cur_case_mp = case_mp

                if getattr(case_mp, "block_mask_bits", None) is not None and isinstance(case_mp.block_mask_bits, np.ndarray) and case_mp.block_mask_bits.size > 0:
                    self.dissolution_mp_subblock()
                    self.precip_mp_subblock(case, case_mp)
                    
                elif case_mp.plane_indexes or case_mp.dissolution_plane_indexes:
                    self.dissolution_mp_subblock()
                    self.precip_mp_subblock(case, case_mp)
                # _dbm = getattr(case_mp, "dissolution_block_mask_bits", None)
                # _has_block_diss = (
                #     _dbm is not None
                #     and isinstance(_dbm, np.ndarray)
                #     and _dbm.size > 0
                #     and np.any(_dbm)
                # )
                # if _has_block_diss:
                #     self.cur_case = case
                #     self.cur_case_mp = case_mp
                #     self.dissolution_mp_subblock()
    
    def dissolve(self):
        for case, case_mp in self.cases.product_case_pairs:
            _dbm = getattr(case_mp, "dissolution_block_mask_bits", None)
            _has_block_diss = (
                _dbm is not None
                and isinstance(_dbm, np.ndarray)
                and _dbm.size > 0
                and np.any(_dbm)
            )
            if case_mp.dissolution_plane_indexes or _has_block_diss:
                self.cur_case = case
                self.cur_case_mp = case_mp
                self.dissolution_mp_subblock()

    def ioz_depth_from_kinetics(self):
        self.curr_time = Config.GENERATED_VALUES.TAU * (self.iteration + 1)
        active_ind = np.where(self.active_times <= self.curr_time)[0]
        return min(np.amax(active_ind), self.furthest_index)

    def ioz_depth_furthest_inward(self):
        oxidant_3d = self.cur_case.oxidant.get_3d_grid()[0]
        # ioz_bound = max x (i) where any oxidant or active particle exists (narrow the domain)
        flat_o = np.flatnonzero(oxidant_3d.ravel(order="F") > 0)
        max_x_ox = int(np.max(flat_o % self.cells_per_axis))
        self.ioz_bound = max(max_x_ox, 0)
        return self.ioz_bound

    def ioz_dissolution_where_prod(self):
        return np.where(self.cur_case.prod_indexes)[0]
