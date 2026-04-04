import os
import numpy as np
import utils
from multiprocessing import shared_memory
from .nes_for_mp import *
from .dissolution_functions import (
    dissolution_subblock_worker,
    get_block_patterns_from_aggregated,
)
from utils.numba_functions import (
    product_counts_upto_bound_from_state,
    product_counts_at_indexes_from_state,
)
from thermodynamics import *
from configuration import Config
from diffusion_3d_mp_example import _partition_domain_z, _partition_gap_z_parallel, _DIRS_6_PACKED
from .nucleation_functions import (
    precip_step_subblock_worker,
)
from .neigh_indexes import ind_formation


class CellularAutomata:
    def __init__(self, cases, utils_inst):
        self.utils = utils_inst
        self.cases = cases
        self.cur_case = None
        self.cur_case_mp = None

        # simulated space parameters
        self.cells_per_axis = Config.N_CELLS_PER_AXIS
        self.cells_per_page = self.cells_per_axis ** 2
        self.matrix_moles_per_page = self.cells_per_page * Config.MATRIX.MOLES_PER_CELL
        self.n_iter = Config.N_ITERATIONS
        self.iteration = None
        self.curr_max_furthest = 0
        self.furthest_index = 0
        self.ioz_bound = 0

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

        self.primary_oxid_numb = Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER
        self.max_inside_neigh_number = 6 * self.primary_oxid_numb
        self.max_block_neigh_number = 7

        self.primary_fetch_ind = []
        self.secondary_fetch_ind = []
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

        self.threshold_inward = Config.THRESHOLD_INWARD
        self.threshold_outward = Config.THRESHOLD_OUTWARD

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
        # Tracks plane-0 concentrations per product and iteration:
        # key=(iteration, product_name), value=(jmatpro_conc, existing_conc, diff)
        self.product_plane0_tracking = {}

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

    def get_combi_ind_standard_v2(self):
        """
        Same logic as get_combi_ind_standard but:
        - Works with current c3d layout (uses .c3d or get_3d_grid()[0] for oxidant/active).
        - ioz_bound = x index of the furthest diffused particle (max over oxidant and active), to narrow the domain.
        - Vectorized: one sum over axes (0,1) per array instead of a Python loop over planes.
        Sets self.ioz_bound, self.comb_indexes.
        """
        oxidant_3d = self.cur_case.oxidant.get_3d_grid()[0]
        active_3d = self.cur_case.active.get_3d_grid()[0]

        # ioz_bound = max x (i) where any oxidant or active particle exists (narrow the domain)
        flat_o = np.flatnonzero(oxidant_3d.ravel(order="F") > 0)
        max_x_ox = int(np.max(flat_o % self.cells_per_axis)) if flat_o.size > 0 else -1
        self.ioz_bound = max(max_x_ox, 0)

        # Sum over (y, z) for each x in one call; result shape (n_i,)
        oxidant = np.sum(oxidant_3d[:self.ioz_bound+1, :, :], axis=(1, 2)).astype(np.uint32)
        active = np.sum(active_3d[:self.ioz_bound+1, :, :], axis=(1, 2)).astype(np.uint32)

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]

        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            self.comb_indexes = oxidant_indexes[indexs]
        else:
            self.comb_indexes = np.array([self.ioz_bound])

    def get_combi_ind_atomic_gamma_prime(self):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles

        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles

        oxidant_c = oxidant_moles / whole_moles
        active_c = active_moles / whole_moles
        product_c = product_moles / whole_moles

        self.gamma_primes = ((((oxidant_c ** 3) * (active_c ** 2)) / Config.SOL_PROD) - 1) /\
                            Config.GENERATED_VALUES.max_gamma_min_one

        where_solub_prod = np.where(self.gamma_primes > 0)[0]
        temp_ind = np.where(product_c[where_solub_prod] < Config.PHASE_FRACTION_LIMIT)[0]
        where_solub_prod = where_solub_prod[temp_ind]

        self.rel_prod_fraction = product_c / Config.PHASE_FRACTION_LIMIT

        self.product_indexes = np.where(product_c > 0)[0]
        prod_left_shift = self.product_indexes - 1
        prod_right_shift = self.product_indexes + 1
        self.product_indexes = np.unique(np.concatenate((self.product_indexes, prod_left_shift, prod_right_shift)))
        temp_ind = np.where((self.product_indexes >= 0) & (self.product_indexes <= self.furthest_index))
        self.product_indexes = self.product_indexes[temp_ind]

        some = np.where(product_c[self.product_indexes] < Config.PHASE_FRACTION_LIMIT)[0]
        self.product_indexes = self.product_indexes[some]

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]
        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            comb_indexes = oxidant_indexes[indexs]
            self.comb_indexes = np.intersect1d(comb_indexes, self.product_indexes)
        else:
            self.comb_indexes = [self.furthest_index]

        self.comb_indexes = np.unique(np.concatenate((self.comb_indexes, where_solub_prod)))

    def get_combi_ind_atomic(self):
        self.ioz_bound = self.get_cur_ioz_bound()

        oxidant = np.array([np.sum(self.cur_case.oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.ioz_bound + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        active = np.array([np.sum(self.cur_case.active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.ioz_bound + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL
        product = self._get_product_counts_upto_bound_for_case(self.cur_case, self.cur_case_mp, self.ioz_bound)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        if self.iteration % Config.STRIDE == 0:
            self.record_prod_per_layer(self.ioz_bound, product_c, np.zeros(product_c.shape))

        self.product_indexes = np.where(product_c <= Config.PHASE_FRACTION_LIMIT)[0]

        self.comb_indexes = self.get_active_oxidant_mutual_indexes(oxidant, active)
        self.comb_indexes = np.intersect1d(self.comb_indexes, self.product_indexes)

    def get_combi_ind_atomic_two_products_gamma(self):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        secondary_active = np.array([np.sum(self.secondary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        secondary_active_moles = secondary_active * Config.ACTIVES.SECONDARY.MOLES_PER_CELL
        secondary_outward_eq_mat_moles = secondary_active * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        secondary_product = np.array([np.sum(self.secondary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        secondary_product_moles = secondary_product * Config.PRODUCTS.SECONDARY.MOLES_PER_CELL
        secondary_product_eq_mat_moles = secondary_product * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles -\
                       secondary_outward_eq_mat_moles - secondary_product_eq_mat_moles

        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles +\
                      secondary_active_moles + secondary_product_moles

        oxidant_c = oxidant_moles / whole_moles
        active_c = active_moles / whole_moles
        secondary_active_c = secondary_active_moles / whole_moles
        product_c = product_moles / whole_moles
        secondary_product_c = secondary_product_moles / whole_moles

        self.gamma_primes = (((((oxidant_c ** 3) * (active_c ** 2)) / Config.SOL_PROD) - 1) /
                             Config.GENERATED_VALUES.max_gamma_min_one)

        where_solub_prod = np.where(self.gamma_primes > 0)[0]
        temp_ind = np.where(product_c[where_solub_prod] < Config.PHASE_FRACTION_LIMIT)[0]
        where_solub_prod = where_solub_prod[temp_ind]

        self.rel_prod_fraction = product_c / Config.PHASE_FRACTION_LIMIT

        self.product_indexes = np.where(product_c > 0)[0]
        prod_left_shift = self.product_indexes - 1
        prod_right_shift = self.product_indexes + 1
        self.product_indexes = np.unique(np.concatenate((self.product_indexes, prod_left_shift, prod_right_shift)))
        temp_ind = np.where((self.product_indexes >= 0) & (self.product_indexes <= self.furthest_index))
        self.product_indexes = self.product_indexes[temp_ind]

        some = np.where(product_c[self.product_indexes] < Config.PHASE_FRACTION_LIMIT)[0]
        self.product_indexes = self.product_indexes[some]

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]
        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            comb_indexes = oxidant_indexes[indexs]
            # self.comb_indexes = comb_indexes
            self.comb_indexes = np.intersect1d(comb_indexes, self.product_indexes)
        else:
            self.comb_indexes = [self.furthest_index]

        self.comb_indexes = np.unique(np.concatenate((self.comb_indexes, where_solub_prod)))

    def get_combi_ind_atomic_two_products(self):
        self.ioz_bound = self.get_cur_ioz_bound()

        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.ioz_bound + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.ioz_bound + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        secondary_active = np.array([np.sum(self.secondary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.ioz_bound + 1)], dtype=np.uint32)
        secondary_active_moles = secondary_active * Config.ACTIVES.SECONDARY.MOLES_PER_CELL
        secondary_outward_eq_mat_moles = secondary_active * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.ioz_bound + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        secondary_product = np.array([np.sum(self.secondary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.ioz_bound + 1)], dtype=np.uint32)
        secondary_product_moles = secondary_product * Config.PRODUCTS.SECONDARY.MOLES_PER_CELL
        secondary_product_eq_mat_moles = secondary_product * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles -\
                       secondary_outward_eq_mat_moles - secondary_product_eq_mat_moles
        less_than_zero = np.where(matrix_moles < 0)[0]
        matrix_moles[less_than_zero] = 0
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles +\
                      secondary_active_moles + secondary_product_moles

        product_c = product_moles / whole_moles
        secondary_product_c = secondary_product_moles / whole_moles

        t_ind_p = np.where(product_c < Config.PRODUCTS.PRIMARY.PHASE_FRACTION_LIMIT)[0]
        t_ind_s = np.where(secondary_product_c < Config.PRODUCTS.SECONDARY.PHASE_FRACTION_LIMIT)[0]

        comb_indexes = []

        self.get_combi_ind_two_products(self.primary_active)
        comb_indexes.append(np.intersect1d(self.comb_indexes, t_ind_p))

        self.get_combi_ind_two_products(self.secondary_active)
        comb_indexes.append(np.intersect1d(self.comb_indexes, t_ind_s))

        self.comb_indexes = comb_indexes

    def get_combi_ind_atomic_with_kinetic(self):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL
        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        self.soll_prod = Config.PROD_INCR_CONST * (Config.GENERATED_VALUES.TAU * (self.iteration + 1))**1.1

        self.cumul_prod.append(product_c[0])
        self.growth_rate.append(self.soll_prod)

        self.product_indexes = np.where((product_c <= Config.PHASE_FRACTION_LIMIT) & (product_c < self.soll_prod))[0]

        self.comb_indexes = self.get_active_oxidant_mutual_indexes(oxidant, active)
        self.comb_indexes = np.intersect1d(self.comb_indexes, self.product_indexes)

    def get_comb_ind_atomic_lut_kin(self):
        self.ioz_bound = self.get_cur_ioz_bound()

        oxidant = np.array([np.sum(self.cases.first.oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.ioz_bound + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.cases.first.active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.ioz_bound + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = self._get_product_counts_upto_bound_for_case(self.cases.first, self.cases.first_mp, self.ioz_bound)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL_TC
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL * \
                               Config.PRODUCTS.PRIMARY.THRESHOLD_OUTWARD

        matrix_moles = (self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles)
        whole_moles = (matrix_moles + oxidant_moles + active_moles + product_moles)

        product_c = product_moles / whole_moles
        times = np.full((self.ioz_bound + 1), self.iteration * Config.GENERATED_VALUES.TAU)
        pos = np.arange(self.ioz_bound + 1) * Config.GENERATED_VALUES.LAMBDA
        curr_look_up = self.KinDATA.get_look_up_data(times, pos)

        if self.iteration % Config.STRIDE == 0:
            self.record_prod_per_layer(self.ioz_bound, product_c, curr_look_up)

        # some = (curr_look_up * (1 - Config.PROD_ERROR))

        self.product_indexes = np.where(product_c < (curr_look_up * (1 - Config.PROD_ERROR)))[0]

        self.comb_indexes = self.get_active_oxidant_mutual_indexes(oxidant, active)
        self.comb_indexes = np.intersect1d(self.comb_indexes, self.product_indexes)
    
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
            task_timeout=3.0,
            max_retries=3,
        )

    def _get_product_counts_upto_bound_for_case(self, case, case_mp, u_bound):
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
        for key in ("PRIMARY", "SECONDARY"):
            cfg = getattr(group, key, None)
            if cfg is None:
                continue
            elem = str(getattr(cfg, "ELEMENT", "None"))
            if elem and elem.lower() != "none":
                out[elem] = cfg
        return out

    def _get_product_cfg_for_case_mp(self, case_mp):
        key = str(getattr(case_mp, "product_key", "PRIMARY"))
        return getattr(Config.PRODUCTS, key, Config.PRODUCTS.PRIMARY)

    def _gather_species_counts_by_element(self, u_bound, species_objs):
        ub = int(u_bound)
        out = {}
        for obj in species_objs:
            elem = str(getattr(obj, "elem_name", ""))
            if not elem:
                continue
            grid = obj.get_3d_grid()[0]
            counts = np.sum(grid[:ub + 1, :, :], axis=(1, 2)).astype(np.float64)
            prev = out.get(elem)
            if prev is None:
                out[elem] = counts
            else:
                out[elem] = prev + counts
        return out

    def _resolve_product_stoich_roles(self, case, case_mp, product_cfg):
        stoich_cfg = product_cfg.STOICH
        stoich = {}
        for k, v in stoich_cfg.items():
            try:
                vv = int(v)
            except (TypeError, ValueError):
                continue
            if vv > 0:
                stoich[str(k)] = vv

        outward = set(str(e) for e in product_cfg.OUTWARD_ELEMENTS)
        inward = set(str(e) for e in product_cfg.INWARD_ELEMENTS)
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

    def get_comb_ind_jmatpro_generic(self):
        self.ioz_bound = self.get_cur_ioz_bound()
        ub = int(self.ioz_bound)

        oxid_cfg = self._cfg_elem_map(Config.OXIDANTS)
        act_cfg = self._cfg_elem_map(Config.ACTIVES)
        oxid_counts = self._gather_species_counts_by_element(ub, self.cases.all_oxidants)
        act_counts = self._gather_species_counts_by_element(ub, self.cases.all_actives)
        total_oxid_counts = np.zeros(ub + 1, dtype=np.float64)
        for v in oxid_counts.values():
            total_oxid_counts += v
        total_act_counts = np.zeros(ub + 1, dtype=np.float64)
        for v in act_counts.values():
            total_act_counts += v

        elem_free_moles = {}
        for elem, counts in oxid_counts.items():
            cfg = oxid_cfg.get(elem)
            elem_free_moles[elem] = counts * cfg.MOLES_PER_CELL
        for elem, counts in act_counts.items():
            cfg = act_cfg.get(elem)
            elem_free_moles[elem] = counts * cfg.MOLES_PER_CELL

        outward_eq_mat_moles = np.zeros(ub + 1, dtype=np.float64)
        for elem, counts in act_counts.items():
            cfg = act_cfg.get(elem)
            outward_eq_mat_moles += counts * cfg.EQ_MATRIX_MOLES_PER_CELL

        product_moles_total = np.zeros(ub + 1, dtype=np.float64)
        product_eq_mat_moles = np.zeros(ub + 1, dtype=np.float64)
        elem_pure_moles = dict(elem_free_moles)
        product_moles_by_identifier = {}
        for case, case_mp in self.cases.product_case_pairs:
            p_cfg = self._get_product_cfg_for_case_mp(case_mp)
            p_counts = self._get_product_counts_upto_bound_for_case(case, case_mp, ub).astype(np.float64)
            p_moles = p_counts * p_cfg.MOLES_PER_CELL_TC
            product_moles_by_identifier[p_cfg.ELEMENT] = p_moles
            product_moles_total += p_moles

            stoich, outward_set, _ = self._resolve_product_stoich_roles(case, case_mp, p_cfg)
            nu_sum = float(sum(stoich.values()))
            for elem, nu in stoich.items():
                frac = float(nu) / nu_sum
                elem_pure_moles[elem] = elem_pure_moles.get(elem, 0.0) + p_moles * frac
                if elem in outward_set:
                    eq_pc = act_cfg[elem].EQ_MATRIX_MOLES_PER_CELL
                    product_eq_mat_moles += p_counts * eq_pc * float(nu)

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + product_moles_total
        for m in elem_free_moles.values():
            whole_moles += m
        
        product_c_by_identifier = {}
        for p_ident, p_moles in product_moles_by_identifier.items():
            product_c_by_identifier[p_ident] = p_moles / whole_moles

        matrix_moles_pure = np.full(ub + 1, self.matrix_moles_per_page, dtype=np.float64)
        for elem, moles in elem_pure_moles.items():
            cfg_a = act_cfg.get(elem)
            t_val = float(getattr(cfg_a, "T", 0.0))
            matrix_moles_pure -= moles * t_val

        matrix_elem = str(getattr(Config.MATRIX, "ELEMENT", "Ni"))
        comp_elems = [e for e in sorted(elem_pure_moles.keys()) if e and e != matrix_elem]
        elements = [matrix_elem] + comp_elems
        compositions = []
        for i in range(ub + 1):
            non_matrix_sum = 0.0
            for elem in comp_elems:
                non_matrix_sum += float(elem_pure_moles[elem][i])
            tot = float(matrix_moles_pure[i] + non_matrix_sum)
            if tot <= 0:
                compositions.append([100.0] + [0.0] * len(comp_elems))
                continue
            row = [float(matrix_moles_pure[i] * 100.0 / tot)]
            for elem in comp_elems:
                row.append(float(elem_pure_moles[elem][i] * 100.0 / tot))
            s = sum(row)
            if s > 100.0:
                scale = 100.0 / s
                row = [v * scale for v in row]
            compositions.append(row)

        plane_indices = np.arange(ub + 1, dtype=np.intp)
        task_ids = self.jmatpro_pool.submit_tasks(compositions, elements=elements)
        task_to_plane = {tid: int(plane_indices[idx]) for idx, tid in enumerate(task_ids)}
        raw_list = self.jmatpro_pool.get_results(task_ids, wait=True, timeout=100000.0)

        for case, case_mp in self.cases.product_case_pairs:
            p_cfg = self._get_product_cfg_for_case_mp(case_mp)
            case_mp.plane_indexes = []
            out_elem = p_cfg.OUTWARD_ELEMENTS[0]
            jm_plane0 = 0.0
            for tid in task_ids:
                phases = raw_list.get(tid, {})
                phased = phases.get(p_cfg.JM_IDENTIFIER)
                if not phased:
                    continue
                sum_non_ox = phased["sum_non_ox"]
                plane_idx = task_to_plane[tid]
                for jm_elem, jm_comp in zip(phased["elements"], phased["composition"]):
                    if jm_elem == out_elem:
                        product_c_jm = (jm_comp / sum_non_ox) * phased["molar_fraction"]
                        if plane_idx == 0:
                            jm_plane0 = float(product_c_jm)
                        if product_c_jm > product_c_by_identifier[p_cfg.ELEMENT][plane_idx]:
                            case_mp.plane_indexes.append(plane_idx)
            if self.iteration is not None:
                existing_plane0 = float(product_c_by_identifier[p_cfg.ELEMENT][0])
                diff_plane0 = jm_plane0 - existing_plane0
                self.product_plane0_tracking[(int(self.iteration), str(p_cfg.ELEMENT))] = (
                    jm_plane0,
                    existing_plane0,
                    diff_plane0,
                )

    def get_comb_ind_jmatpro(self):
        """Single active, single oxidant only (no secondary elements)."""
        self.ensure_jmatpro_pool()
        self.ioz_bound = self.get_cur_ioz_bound()

        # Inward/outward now come from diffusion read grids.
        oxidant_3d = self.cur_case.oxidant.get_3d_grid()[0]
        oxidant = np.sum(oxidant_3d[:self.ioz_bound + 1, :, :], axis=(1, 2)).astype(np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active_3d = self.cur_case.active.get_3d_grid()[0]
        active = np.sum(active_3d[:self.ioz_bound + 1, :, :], axis=(1, 2)).astype(np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = self._get_product_counts_upto_bound_for_case(self.cur_case, self.cur_case_mp, self.ioz_bound)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL_TC
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL * \
                               Config.PRODUCTS.PRIMARY.THRESHOLD_OUTWARD

        matrix_moles = (self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles)
        whole_moles = (matrix_moles + oxidant_moles + active_moles + product_moles)

        product_c = product_moles / whole_moles

        oxidant_pure_moles = (oxidant_moles + (product_moles * 3/5))
        active_pure_moles = active_moles + (product_moles * 2/5)
        active_pure_eq_mat_moles = active_pure_moles * Config.ACTIVES.PRIMARY.T

        matrix_moles_pure = self.matrix_moles_per_page - active_pure_eq_mat_moles
        whole_moles_pure = matrix_moles_pure + oxidant_pure_moles + active_pure_moles

        oxidant_pure_c = oxidant_pure_moles * 100 / whole_moles_pure
        active_pure_c = active_pure_moles * 100 / whole_moles_pure

        elements = [
            getattr(Config.MATRIX, "ELEMENT", "Ni"),
            self.cur_case.active.elem_name,
            self.cur_case.oxidant.elem_name,
        ]
        compositions = []
        for i in range(len(active_pure_c)):
            a_i = float(active_pure_c[i])
            o_i = float(oxidant_pure_c[i])
            base = 100.0 - a_i - o_i
            if base < 0:
                base = 0.0
                total = a_i + o_i
                if total > 100.0:
                    scale = 100.0 / total
                    compositions.append([base, a_i * scale, o_i * scale])
                else:
                    compositions.append([base, a_i, o_i])
            else:
                compositions.append([base, a_i, o_i])

        task_ids = self.jmatpro_pool.submit_tasks(compositions, elements=elements)
        raw_list = self.jmatpro_pool.get_results(task_ids, wait=True, timeout=100000.0)

        # Preserve composition index: result[i] must match composition[i] (task_ids[i])
        def _m2o3_fraction(raw):
            m2o3 = raw.get("M2O3") if isinstance(raw, dict) else None
            if m2o3 is None:
                return 0.0
            if isinstance(m2o3, dict):
                return float(m2o3.get("molar_fraction", 0.0))
            return float(m2o3)
        curr_look_up = np.array([_m2o3_fraction(raw_list.get(tid, {})) for tid in task_ids])

        t_ind_p = np.where(curr_look_up > 0)[0]
        t_ind_z = np.where(curr_look_up == 0)[0]
        primary_error = (curr_look_up[t_ind_p] - product_c[t_ind_p]) / curr_look_up[t_ind_p]
        primary_pos_ind = t_ind_p[np.where(primary_error > Config.PROD_ERROR)[0]]

        coef_ind = np.where(primary_error < -Config.PROD_ERROR)[0]
        primary_neg_ind = t_ind_p[coef_ind]
        adj_coeff_neg = primary_error[coef_ind] * -1
        d_ind = t_ind_z[np.where(product_c[t_ind_z] > 0)[0]]

        if len(primary_pos_ind) > 0:
            oxidant_indexes = np.where(oxidant > 0)[0]
            active_indexes = np.where(active > 0)[0]
            min_act = active_indexes.min(initial=self.cells_per_axis)
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            self.comb_indexes = oxidant_indexes[indexs]
            self.comb_indexes = np.intersect1d(primary_pos_ind, self.comb_indexes)
        else:
            self.comb_indexes = []

            # if len(self.comb_indexes) > 0:
                # self.cases.reaccumulate_products(self.cur_case)
                # self.precip_mp()

        # if len(primary_neg_ind) > 0 or len(d_ind) > 0 or len(primary_pos_ind) > 0:
        #     self.comb_indexes = np.concatenate((primary_neg_ind, d_ind, primary_pos_ind))
        #     adj_coeff = np.concatenate((adj_coeff_neg, np.ones(len(d_ind)), np.zeros(len(primary_pos_ind))))
        #     self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes, adj_coeff)
        #     self.decomposition_intrinsic()

    def get_combi_ind_atomic_with_kinetic_and_KP(self):
        self.ioz_bound = self.get_cur_ioz_bound()

        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.ioz_bound + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.ioz_bound + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL
        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.ioz_bound + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        less_than_zero = np.where(matrix_moles < 0)[0]
        matrix_moles[less_than_zero] = 0
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        powers = self.powers[np.arange(self.ioz_bound + 1)]
        soll_prod = Config.PROD_INCR_CONST * (self.curr_time - self.active_times[:self.ioz_bound + 1]) ** powers
        # soll_prod = Config.PROD_INCR_CONST * (self.curr_time - self.active_times[:ioz_bound + 1]) ** 1.1

        self.diffs = product_c - soll_prod

        if self.iteration % Config.STRIDE == 0:
            self.record_prod_per_layer(self.ioz_bound, product_c, soll_prod)

        # self.product_indexes = np.where((product_c <= Config.PHASE_FRACTION_LIMIT) & (self.diffs <= 0))[0]
        # self.product_indexes = np.where(product_c <= Config.PHASE_FRACTION_LIMIT)[0]
        self.product_indexes = np.where(self.diffs <= 0)[0]

        self.comb_indexes = self.get_active_oxidant_mutual_indexes(oxidant, active)
        self.comb_indexes = np.intersect1d(self.comb_indexes, self.product_indexes)

    def get_combi_ind_atomic_solub_prod_test(self):
        oxidant = np.array([np.sum(self.cur_case.oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.cur_case.active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = self._get_product_counts_upto_bound_for_case(self.cur_case, self.cur_case_mp, self.furthest_index)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL_TC
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL *\
                               Config.PRODUCTS.PRIMARY.THRESHOLD_OUTWARD

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles

        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles

        oxidant_c = oxidant_moles / whole_moles
        active_c = active_moles / whole_moles

        # self.gamma_primes = (((((oxidant_c ** 3) * (active_c ** 2)) / Config.SOL_PROD) - 1) /
        #                      Config.GENERATED_VALUES.max_gamma_min_one)

        k = (oxidant_c ** 3) * (active_c ** 2)

        # where_solub_prod = np.where(self.gamma_primes > 0)[0]

        self.comb_indexes = np.where(k >= Config.SOL_PROD)[0]

        # oxidant_indexes = np.where(oxidant > 0)[0]
        # active_indexes = np.where(active > 0)[0]
        # min_act = active_indexes.min(initial=self.cells_per_axis)
        # if min_act < self.cells_per_axis:
        #     indexs = np.where(oxidant_indexes >= min_act - 1)[0]
        #     self.comb_indexes = oxidant_indexes[indexs]
        # else:
        #     self.comb_indexes = [self.furthest_index]

        # self.comb_indexes = np.intersect1d(self.comb_indexes, where_solub_prod)

    def get_combi_ind_atomic_opt_for_growth(self):

        w_int = np.where(self.product_x_not_stab[:self.furthest_index + 1])[0]

        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind in w_int], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind in w_int], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind in w_int], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        self.nucleation_indexes = w_int[np.where(product_c <= Config.PHASE_FRACTION_LIMIT)[0]]

        stab_prod_ind = np.where(product_c > Config.PHASE_FRACTION_LIMIT)[0]
        self.product_x_not_stab[w_int[stab_prod_ind]] = False

        # self.product_indexes = np.where(product_c > 0)[0]
        # prod_left_shift = self.product_indexes - 1
        # prod_right_shift = self.product_indexes + 1
        # self.product_indexes = np.unique(np.concatenate((self.product_indexes, prod_left_shift, prod_right_shift)))
        # temp_ind = np.where((self.product_indexes >= 0) & (self.product_indexes <= self.furthest_index))
        # self.product_indexes = self.product_indexes[temp_ind]

        # some = np.where((product_c[self.product_indexes] < self.param["phase_fraction_lim"]) & (product_c[self.product_indexes] > 0))[0]
        # some = np.where(product_c[self.product_indexes] < self.param["phase_fraction_lim"])[0]
        # self.product_indexes = self.product_indexes[some]

        act_ox_mutual_ind = self.get_active_oxidant_mutual_indexes(oxidant, active)
        self.comb_indexes = np.intersect1d(act_ox_mutual_ind, self.nucleation_indexes)

        # oxidant_indexes = np.where(oxidant > 0)[0]
        # active_indexes = np.where(active > 0)[0]
        # min_act = active_indexes.min(initial=self.cells_per_axis)
        # if min_act < self.cells_per_axis:
        #     indexs = np.where(oxidant_indexes >= min_act - 1)[0]
        #     comb_indexes = oxidant_indexes[indexs]
        #     self.comb_indexes = np.intersect1d(comb_indexes, self.product_indexes)
        # else:
        #     self.comb_indexes = [self.furthest_index]

    def calc_stable_products(self):
        self.ioz_bound = self.get_cur_ioz_bound()
        self.ensure_td_lookup()

        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.ioz_bound + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.ioz_bound + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        secondary_active = np.array([np.sum(self.secondary_active.c3d[:, :, plane_ind]) for plane_ind
                                     in range(self.ioz_bound + 1)], dtype=np.uint32)
        secondary_active_moles = secondary_active * Config.ACTIVES.SECONDARY.MOLES_PER_CELL
        secondary_outward_eq_mat_moles = secondary_active * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.ioz_bound + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL_TC
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        secondary_product = np.array([np.sum(self.secondary_product.c3d[:, :, plane_ind]) for plane_ind
                                      in range(self.ioz_bound + 1)], dtype=np.uint32)
        secondary_product_moles = secondary_product * Config.PRODUCTS.SECONDARY.MOLES_PER_CELL_TC
        secondary_product_eq_mat_moles = secondary_product * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = (self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles -
                        secondary_outward_eq_mat_moles - secondary_product_eq_mat_moles)
        neg_ind = np.where(matrix_moles < 0)[0]
        matrix_moles[neg_ind] = 0
        whole_moles = (matrix_moles + oxidant_moles + active_moles + product_moles + secondary_active_moles +
                       secondary_product_moles)

        product_c = product_moles / whole_moles
        secondary_product_c = secondary_product_moles / whole_moles

        # oxidant_pure = oxidant + product + secondary_product
        # oxidant_pure_moles = oxidant_pure * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        #
        # active_pure = active + product
        # active_pure_moles = active_pure * self.param["active_element"]["primary"]["moles_per_cell"]
        # active_pure_eq_mat_moles = active_pure * self.param["active_element"]["primary"]["eq_matrix_moles_per_cell"]
        #
        # secondary_active_pure = secondary_active + secondary_product
        # secondary_active_pure_moles = secondary_active_pure * self.param["active_element"]["secondary"]["moles_per_cell"]
        # secondary_active_pure_eq_mat_moles = secondary_active_pure * self.param["active_element"]["secondary"]["eq_matrix_moles_per_cell"]
        #
        # matrix_moles_pure = self.matrix_moles_per_page - active_pure_eq_mat_moles - secondary_active_pure_eq_mat_moles
        # whole_moles_pure = matrix_moles_pure + oxidant_pure_moles + active_pure_moles + secondary_active_pure_moles
        #
        # oxidant_pure_c = oxidant_pure_moles / whole_moles_pure
        # active_pure_c = active_pure_moles / whole_moles_pure
        # secondary_active_pure_c = secondary_active_pure_moles / whole_moles_pure

        oxidant_pure_moles = oxidant_moles + product_moles * 3 + secondary_product_moles * 3

        active_pure_moles = active_moles + product_moles * 2
        active_pure_eq_mat_moles = active_pure_moles * Config.ACTIVES.PRIMARY.T
        secondary_active_pure_moles = secondary_active_moles + secondary_product_moles * 2
        secondary_active_pure_eq_mat_moles = secondary_active_pure_moles * Config.ACTIVES.SECONDARY.T

        matrix_moles_pure = self.matrix_moles_per_page - active_pure_eq_mat_moles - secondary_active_pure_eq_mat_moles
        neg_ind = np.where(matrix_moles_pure < 0)[0]
        matrix_moles_pure[neg_ind] = 0
        whole_moles_pure = matrix_moles_pure + oxidant_pure_moles + active_pure_moles + secondary_active_pure_moles

        oxidant_pure_c = oxidant_pure_moles * 100 / whole_moles_pure
        active_pure_c = active_pure_moles * 100 / whole_moles_pure
        secondary_active_pure_c = secondary_active_pure_moles * 100 / whole_moles_pure

        self.curr_look_up = self.TdDATA.get_look_up_data(active_pure_c, secondary_active_pure_c, oxidant_pure_c)

        primary_diff = self.curr_look_up[0] - product_c
        primary_pos_ind = np.where(primary_diff >= 0)[0]
        primary_neg_ind = np.where(primary_diff < 0)[0]

        secondary_diff = self.curr_look_up[1] - secondary_product_c
        secondary_pos_ind = np.where(secondary_diff >= 0)[0]
        secondary_neg_ind = np.where(secondary_diff < 0)[0]

        self.cur_case = self.cases.first
        self.cur_case_mp = self.cases.first_mp
        if len(primary_pos_ind) > 0:
            oxidant_indexes = np.where(oxidant > 0)[0]
            active_indexes = np.where(active > 0)[0]
            min_act = active_indexes.min(initial=self.cells_per_axis)
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            self.comb_indexes = oxidant_indexes[indexs]

            self.comb_indexes = np.intersect1d(primary_pos_ind, self.comb_indexes)

            if len(self.comb_indexes) > 0:
                # self.cur_case.fix_init_precip_func_ref(self.furthest_index)
                self.precip_mp()
                self.decomposition_intrinsic()

        if len(primary_neg_ind) > 0:
            self.comb_indexes = primary_neg_ind
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes, np.ones(len(self.comb_indexes)))
            self.decomposition_intrinsic()
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes,
                                                                        np.zeros(len(self.comb_indexes)))

        self.cur_case = self.cases.second
        self.cur_case_mp = self.cases.second_mp
        if len(secondary_pos_ind) > 0:
            self.get_combi_ind_two_products(self.secondary_active)
            self.comb_indexes = np.intersect1d(secondary_pos_ind, self.comb_indexes)

            if len(self.comb_indexes) > 0:
                # self.cur_case.fix_init_precip_func_ref(self.furthest_index)
                self.precip_mp()
                self.decomposition_intrinsic()

        if len(secondary_neg_ind) > 0:
            self.comb_indexes = secondary_neg_ind
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes,
                                                                        np.ones(len(self.comb_indexes)))
            self.decomposition_intrinsic()
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes,
                                                                        np.zeros(len(self.comb_indexes)))

    def calc_stable_products_all(self):
        self.ioz_bound = self.get_cur_ioz_bound()
        self.ensure_td_lookup()

        oxidant = np.array([np.sum(self.cases.first.oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.ioz_bound + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.cases.first.active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.ioz_bound + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        secondary_active = np.array([np.sum(self.cases.second.active.c3d[:, :, plane_ind]) for plane_ind
                                     in range(self.ioz_bound + 1)], dtype=np.uint32)
        secondary_active_moles = secondary_active * Config.ACTIVES.SECONDARY.MOLES_PER_CELL
        secondary_outward_eq_mat_moles = secondary_active * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        product = self._get_product_counts_upto_bound_for_case(self.cases.first, self.cases.first_mp, self.ioz_bound)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL_TC
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL * \
                               Config.PRODUCTS.PRIMARY.THRESHOLD_OUTWARD

        secondary_product = self._get_product_counts_upto_bound_for_case(self.cases.second, self.cases.second_mp, self.ioz_bound)
        secondary_product_moles = secondary_product * Config.PRODUCTS.SECONDARY.MOLES_PER_CELL_TC
        secondary_product_eq_mat_moles = secondary_product * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL * \
                                         Config.PRODUCTS.SECONDARY.THRESHOLD_OUTWARD

        ternary_product = self._get_product_counts_upto_bound_for_case(self.cases.third, self.cases.third_mp, self.ioz_bound)
        ternary_product_moles = ternary_product * Config.PRODUCTS.TERNARY.MOLES_PER_CELL_TC
        ternary_product_eq_mat_moles = (ternary_product * ((Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL *
                                                            Config.PRODUCTS.TERNARY.THRESHOLD_OUTWARD) + Config.PRODUCTS.TERNARY.MOLES_PER_CELL))

        quaternary_product = self._get_product_counts_upto_bound_for_case(self.cases.fourth, self.cases.fourth_mp, self.ioz_bound)
        quaternary_product_moles = quaternary_product * Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL_TC
        quaternary_product_eq_mat_moles = (quaternary_product * ((Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL *
                                                                  Config.PRODUCTS.QUATERNARY.THRESHOLD_OUTWARD) + Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL))

        quint_product = self._get_product_counts_upto_bound_for_case(self.cases.fifth, self.cases.fifth_mp, self.ioz_bound)
        quint_eq_mat_moles = quint_product * Config.PRODUCTS.QUINT.MOLES_PER_CELL
        quint_product_moles = quint_product * Config.PRODUCTS.QUINT.MOLES_PER_CELL_TC

        matrix_moles = (self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles -
                        secondary_outward_eq_mat_moles - secondary_product_eq_mat_moles - ternary_product_eq_mat_moles -
                        quaternary_product_eq_mat_moles - quint_eq_mat_moles)

        whole_moles = (matrix_moles + oxidant_moles + active_moles + product_moles + secondary_active_moles +
                       secondary_product_moles + ternary_product_moles + quaternary_product_moles + quint_product_moles)

        product_c = product_moles / whole_moles
        secondary_product_c = secondary_product_moles / whole_moles
        ternary_product_c = ternary_product_moles / whole_moles
        quaternary_product_c = quaternary_product_moles / whole_moles
        quint_product_c = quint_product_moles / whole_moles

        oxidant_pure_moles = (oxidant_moles + (product_moles * 3/5) + (secondary_product_moles * 3/5) +
                              (ternary_product_moles * 4/7) + (quaternary_product_moles * 4/7) + (quint_product_moles * 1/2))

        active_pure_moles = active_moles + (product_moles * 2/5) + (ternary_product_moles * 2/7)
        active_pure_eq_mat_moles = active_pure_moles * Config.ACTIVES.PRIMARY.T

        secondary_active_pure_moles = secondary_active_moles + (secondary_product_moles * 2/5) + (quaternary_product_moles * 2/7)
        secondary_active_pure_eq_mat_moles = secondary_active_pure_moles * Config.ACTIVES.SECONDARY.T

        matrix_moles_pure = self.matrix_moles_per_page - active_pure_eq_mat_moles - secondary_active_pure_eq_mat_moles
        whole_moles_pure = matrix_moles_pure + oxidant_pure_moles + active_pure_moles + secondary_active_pure_moles

        oxidant_pure_c = oxidant_pure_moles * 100 / whole_moles_pure
        active_pure_c = active_pure_moles * 100 / whole_moles_pure
        secondary_active_pure_c = secondary_active_pure_moles * 100 / whole_moles_pure

        curr_look_up = self.TdDATA.get_look_up_data(active_pure_c, secondary_active_pure_c, oxidant_pure_c)

        t_ind_p = np.where(curr_look_up[0] > 0)[0]
        t_ind_z = np.where(curr_look_up[0] == 0)[0]
        primary_error = (curr_look_up[0][t_ind_p] - product_c[t_ind_p]) / curr_look_up[0][t_ind_p]
        primary_pos_ind = t_ind_p[np.where(primary_error > Config.PROD_ERROR)[0]]

        coef_ind = np.where(primary_error < -Config.PROD_ERROR)[0]
        primary_neg_ind = t_ind_p[coef_ind]
        adj_coeff_neg = primary_error[coef_ind] * -1

        d_ind = t_ind_z[np.where(product_c[t_ind_z] > 0)[0]]

        self.cur_case = self.cases.first
        self.cur_case_mp = self.cases.first_mp
        if len(primary_pos_ind) > 0:
            oxidant_indexes = np.where(oxidant > 0)[0]
            active_indexes = np.where(active > 0)[0]
            min_act = active_indexes.min(initial=self.cells_per_axis)
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            self.comb_indexes = oxidant_indexes[indexs]
            self.comb_indexes = np.intersect1d(primary_pos_ind, self.comb_indexes)

            if len(self.comb_indexes) > 0:
                self.cases.reaccumulate_products(self.cur_case)
                self.precip_mp()

        if len(primary_neg_ind) > 0 or len(d_ind) > 0 or len(primary_pos_ind) > 0:
            self.comb_indexes = np.concatenate((primary_neg_ind, d_ind, primary_pos_ind))
            adj_coeff = np.concatenate((adj_coeff_neg, np.ones(len(d_ind)), np.zeros(len(primary_pos_ind))))
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes, adj_coeff)
            self.decomposition_intrinsic()

        t_ind_p = np.where(curr_look_up[1] > 0)[0]
        t_ind_z = np.where(curr_look_up[1] == 0)[0]
        secondary_error = (curr_look_up[1][t_ind_p] - secondary_product_c[t_ind_p]) / curr_look_up[1][t_ind_p]
        secondary_pos_ind = t_ind_p[np.where(secondary_error > Config.PROD_ERROR)[0]]

        coef_ind = np.where(secondary_error < -Config.PROD_ERROR)[0]
        secondary_neg_ind = t_ind_p[coef_ind]
        adj_coeff_neg = secondary_error[coef_ind] * -1

        d_ind = t_ind_z[np.where(secondary_product_c[t_ind_z] > 0)[0]]

        self.cur_case = self.cases.second
        self.cur_case_mp = self.cases.second_mp
        if len(secondary_pos_ind) > 0:
            self.get_combi_ind_two_products()
            self.comb_indexes = np.intersect1d(secondary_pos_ind, self.comb_indexes)

            if len(self.comb_indexes) > 0:
                self.cases.reaccumulate_products(self.cur_case)
                self.precip_mp()

        if len(secondary_neg_ind) > 0 or len(d_ind) > 0 or len(secondary_pos_ind) > 0:
            self.comb_indexes = np.concatenate((secondary_neg_ind, d_ind, secondary_pos_ind))
            adj_coeff = np.concatenate((adj_coeff_neg, np.ones(len(d_ind)), np.zeros(len(secondary_pos_ind))))
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes, adj_coeff)
            self.decomposition_intrinsic()


        t_ind_p = np.where(curr_look_up[2] > 0)[0]
        t_ind_z = np.where(curr_look_up[2] == 0)[0]
        ternary_error = (curr_look_up[2][t_ind_p] - ternary_product_c[t_ind_p]) / curr_look_up[2][t_ind_p]
        ternary_pos_ind = t_ind_p[np.where(ternary_error > Config.PROD_ERROR)[0]]

        coef_ind = np.where(ternary_error < -Config.PROD_ERROR)[0]
        ternary_neg_ind = t_ind_p[coef_ind]
        adj_coeff_neg = ternary_error[coef_ind] * -1

        d_ind = t_ind_z[np.where(ternary_product_c[t_ind_z] > 0)[0]]

        self.cur_case = self.cases.third
        self.cur_case_mp = self.cases.third_mp
        if len(ternary_pos_ind) > 0:
            self.get_combi_ind_two_products()
            self.comb_indexes = np.intersect1d(ternary_pos_ind, self.comb_indexes)

            if len(self.comb_indexes) > 0:
                self.cases.reaccumulate_products(self.cur_case)
                self.precip_mp()

        if len(ternary_neg_ind) > 0 or len(d_ind) > 0 or len(ternary_pos_ind) > 0:
            self.comb_indexes = np.concatenate((ternary_neg_ind, d_ind, ternary_pos_ind))
            adj_coeff = np.concatenate((adj_coeff_neg, np.ones(len(d_ind)), np.zeros(len(ternary_pos_ind))))
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes, adj_coeff)
            self.decomposition_intrinsic()


        t_ind_p = np.where(curr_look_up[3] > 0)[0]
        t_ind_z = np.where(curr_look_up[3] == 0)[0]
        quaternary_error = (curr_look_up[3][t_ind_p] - quaternary_product_c[t_ind_p]) / curr_look_up[3][t_ind_p]
        quaternary_pos_ind = t_ind_p[np.where(quaternary_error > Config.PROD_ERROR)[0]]

        coef_ind = np.where(quaternary_error < -Config.PROD_ERROR)[0]
        quaternary_neg_ind = t_ind_p[coef_ind]
        adj_coeff_neg = quaternary_error[coef_ind] * -1

        d_ind = t_ind_z[np.where(quaternary_product_c[t_ind_z] > 0)[0]]

        self.cur_case = self.cases.fourth
        self.cur_case_mp = self.cases.fourth_mp
        if len(quaternary_pos_ind) > 0:
            self.get_combi_ind_two_products()
            self.comb_indexes = np.intersect1d(quaternary_pos_ind, self.comb_indexes)

            if len(self.comb_indexes) > 0:
                self.cases.reaccumulate_products(self.cur_case)
                self.precip_mp()

        if len(quaternary_neg_ind) > 0 or len(d_ind) > 0 or len(quaternary_pos_ind) > 0:
            self.comb_indexes = np.concatenate((quaternary_neg_ind, d_ind, quaternary_pos_ind))
            adj_coeff = np.concatenate((adj_coeff_neg, np.ones(len(d_ind)), np.zeros(len(quaternary_pos_ind))))
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes, adj_coeff)
            self.decomposition_intrinsic()


        t_ind_p = np.where(curr_look_up[4] > 0)[0]
        t_ind_z = np.where(curr_look_up[4] == 0)[0]
        quint_error = (curr_look_up[4][t_ind_p] - quint_product_c[t_ind_p]) / curr_look_up[4][t_ind_p]
        quint_pos_ind = t_ind_p[np.where(quint_error > Config.PROD_ERROR)[0]]

        coef_ind = np.where(quint_error < -Config.PROD_ERROR)[0]
        quint_neg_ind = t_ind_p[coef_ind]
        adj_coeff_neg = quint_error[coef_ind] * -1

        d_ind = t_ind_z[np.where(quint_product_c[t_ind_z] > 0)[0]]

        # secondary_diff = curr_look_up[1] - secondary_product_c
        # secondary_pos_ind = np.where(secondary_diff > err)[0]
        # secondary_neg_ind = np.where(secondary_diff < 0)[0]
        #
        # ternary_diff = curr_look_up[2] - ternary_product_c
        # ternary_pos_ind = np.where(ternary_diff > err)[0]
        # ternary_neg_ind = np.where(ternary_diff < 0)[0]
        #
        # quaternary_diff = curr_look_up[3] - quaternary_product_c
        # quaternary_pos_ind = np.where(quaternary_diff > err)[0]
        # quaternary_neg_ind = np.where(quaternary_diff < 0)[0]
        #
        # quint_diff = curr_look_up[4] - quint_product_c
        # quint_pos_ind = np.where(quint_diff > err)[0]
        # quint_neg_ind = np.where(quint_diff < 0)[0]

        self.cur_case = self.cases.fifth
        self.cur_case_mp = self.cases.fifth_mp
        if len(quint_pos_ind) > 0:
            self.comb_indexes = quint_pos_ind

            if len(self.comb_indexes) > 0:
                self.cases.reaccumulate_products(self.cur_case)
                self.precip_mp()

        if len(quint_neg_ind) > 0 or len(d_ind) > 0 or len(quint_pos_ind) > 0:
            self.comb_indexes = np.concatenate((quint_neg_ind, d_ind, quint_pos_ind))
            adj_coeff = np.concatenate((adj_coeff_neg, np.ones(len(d_ind)), np.zeros(len(quint_pos_ind))))
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes, adj_coeff)
            self.decomposition_intrinsic()

    def precipitation_with_td(self):
        self.furthest_index = self.cases.first.oxidant.calc_furthest_index()

        if self.furthest_index >= self.curr_max_furthest:
            self.curr_max_furthest = self.furthest_index

        self.cases.first.oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            self.cases.first.active.transform_to_3d(self.curr_max_furthest)
            self.cases.second.active.transform_to_3d(self.curr_max_furthest)

        self.calc_stable_products_all()
        self.cases.first.oxidant.transform_to_descards()

    def precipitation_current_case(self):
        # Only one oxidant and one active elements exist. Only one product can be created
        self.furthest_index = self.cur_case.oxidant.calc_furthest_index()
        self.cur_case.oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            if self.furthest_index >= self.curr_max_furthest:
                self.curr_max_furthest = self.furthest_index
            self.cur_case.active.transform_to_3d(self.curr_max_furthest)

        self.get_combi_ind()

        if len(self.comb_indexes) > 0:
            self.precip_mp()

        self.cur_case.oxidant.transform_to_descards()

    def precipitation_second_case(self):
        # Only one oxidant and one active elements exist. Only one product can be created
        self.furthest_index = self.primary_oxidant.calc_furthest_index()
        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            if self.furthest_index >= self.curr_max_furthest:
                self.curr_max_furthest = self.furthest_index
            self.primary_active.transform_to_3d(self.curr_max_furthest)
            self.secondary_active.transform_to_3d(self.curr_max_furthest)

        self.get_combi_ind()

        saved_ind = self.comb_indexes[1]

        if len(self.comb_indexes[0]) > 0:
            self.comb_indexes = self.comb_indexes[0]
            self.cur_case = self.cases.first
            self.cur_case_mp = self.cases.first_mp

            self.precip_mp()

        if len(saved_ind) > 0:
            self.comb_indexes = saved_ind
            self.cur_case = self.cases.second
            self.cur_case_mp = self.cases.second_mp

            self.precip_mp()

        self.primary_oxidant.transform_to_descards()

    def precipitation_growth_test(self):
        # created to test how growth function ang probabilities work
        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            self.primary_active.transform_to_3d(self.cells_per_axis)

        self.comb_indexes = np.where(self.product_x_nzs)[0]
        prod_left_shift = self.comb_indexes - 1
        prod_right_shift = self.comb_indexes + 1
        self.comb_indexes = np.unique(np.concatenate((self.comb_indexes, prod_left_shift, prod_right_shift)))

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.comb_indexes], dtype=np.uint32)
        product_conc = product / (self.cells_per_page * self.primary_oxid_numb)

        # middle_ind = np.where(self.comb_indexes == self.mid_point_coord)[0]
        # rel_phase_fraction_for_all = product_conc[middle_ind] / self.param["phase_fraction_lim"]

        some = np.where(product_conc < Config.PHASE_FRACTION_LIMIT)[0]

        self.comb_indexes = self.comb_indexes[some]

        # rel_product_fractions = product_conc[some] / self.param["phase_fraction_lim"]
        # rel_product_fractions[:] = rel_phase_fraction_for_all

        if len(self.comb_indexes) > 0:
            # self.nucl_prob.adapt_probabilities(self.comb_indexes, rel_product_fractions)
            self.cases.first.fix_init_precip_func_ref(self.cells_per_axis)
            self.precip_step()

        self.primary_oxidant.transform_to_descards()

    def precipitation_growth_test_with_p1(self):
        # in this case single probability for growth were given, if at least one product neighbour then nucleation with
        # P1. the probability functions was were adapted accordingly.
        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            self.primary_active.transform_to_3d(self.cells_per_axis)

        self.comb_indexes = np.where(self.product_x_nzs)[0]
        prod_left_shift = self.comb_indexes - 1
        prod_right_shift = self.comb_indexes + 1
        self.comb_indexes = np.unique(np.concatenate((self.comb_indexes, prod_left_shift, prod_right_shift)))

        u_bound = self.comb_indexes.max()
        l_bound = self.comb_indexes.min()

        # product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
        #                     in self.comb_indexes], dtype=np.uint32)
        # product_conc = product / (self.cells_per_page * self.primary_oxid_numb)

        # middle_ind = np.where(self.comb_indexes == self.mid_point_coord)[0]
        # rel_phase_fraction_for_all = product_conc[middle_ind] / self.param["phase_fraction_lim"]

        # some = np.where(product_conc < self.param["phase_fraction_lim"])[0]

        # self.comb_indexes = self.comb_indexes[some]

        # rel_product_fractions = product_conc[some] / self.param["phase_fraction_lim"]
        # rel_product_fractions[:] = rel_phase_fraction_for_all

        if len(self.comb_indexes) > 0:
            self.cases.first.fix_init_precip_func_ref(u_bound, l_bound=l_bound)
            self.precip_step()

        self.primary_oxidant.transform_to_descards()

    def precipitation_first_case_no_growth(self):
        # Only one oxidant and one active elements exist. Only one product can be created
        self.furthest_index = self.primary_oxidant.calc_furthest_index()
        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            if self.furthest_index >= self.curr_max_furthest:
                self.curr_max_furthest = self.furthest_index
            self.primary_active.transform_to_3d(self.curr_max_furthest)

        self.get_combi_ind()

        if len(self.comb_indexes) > 0:
            # self.cur_case = self.cases.first
            self.precip_step()
        self.primary_oxidant.transform_to_descards()

    def precipitation_0_cells_no_growth_solub_prod_test(self):
        """
        Created only for tests of the solubility product probability function
        """
        # Only one oxidant and one active elements exist. Only one product can be created
        self.furthest_index = self.primary_oxidant.calc_furthest_index()
        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            if self.furthest_index >= self.curr_max_furthest:
                self.curr_max_furthest = self.furthest_index
            self.primary_active.transform_to_3d(self.curr_max_furthest)

        self.get_combi_ind_atomic_solub_prod_test()

        if len(self.comb_indexes) > 0:
            self.nucl_prob.adapt_probabilities(self.comb_indexes, self.gamma_primes[self.comb_indexes])
            self.precip_step_no_growth_solub_prod_test()
        self.primary_oxidant.transform_to_descards()

    def dissolution_atomic_stop_if_stable(self):
        self.product_indexes = np.where(self.cur_case.prod_indexes)[0]
        where_not_stab = np.where(self.cur_case.product_ind_not_stab)[0]
        self.product_indexes = np.intersect1d(self.product_indexes, where_not_stab)

        new_stab_count = np.count_nonzero(~self.cur_case.product_ind_not_stab)
        if new_stab_count > self.prev_stab_count:
            self.prev_stab_count = new_stab_count
            print("stable now at: ", np.nonzero(~self.cur_case.product_ind_not_stab)[0])

        self.cur_case.oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.cur_case.oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        self.cur_case.oxidant.transform_to_descards()

        active = np.array([np.sum(self.cur_case.active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = self._get_product_counts_at_indexes_for_case(self.cur_case, self.cur_case_mp, self.product_indexes)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        temp_ind = np.where(product == 0)[0]
        self.cur_case.prod_indexes[self.product_indexes[temp_ind]] = False
        product_c = np.delete(product_c, temp_ind)
        self.comb_indexes = np.delete(self.product_indexes, temp_ind)

        temp_ind = np.where(product_c > Config.PHASE_FRACTION_LIMIT)[0]

        self.cur_case.product_ind_not_stab[self.comb_indexes[temp_ind]] = False
        self.comb_indexes = np.delete(self.comb_indexes, temp_ind)

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()

    def dissolution_atomic_stop_if_stable_two_products(self):
        p_product_indexes = np.where(self.cases.first.prod_indexes)[0]
        p_where_not_stab = np.where(self.cases.first.product_ind_not_stab)[0]
        p_product_indexes = np.intersect1d(p_product_indexes, p_where_not_stab)

        s_product_indexes = np.where(self.cases.second.prod_indexes)[0]
        s_where_not_stab = np.where(self.cases.second.product_ind_not_stab)[0]
        s_product_indexes = np.intersect1d(s_product_indexes, s_where_not_stab)

        self.product_indexes = np.union1d(p_product_indexes, s_product_indexes)
        self.comb_indexes = []
        # new_stab_count = np.count_nonzero(~self.cur_case.product_ind_not_stab)
        # if new_stab_count > self.prev_stab_count:
        #     self.prev_stab_count = new_stab_count
        #     print("stable now at: ", np.nonzero(~self.cur_case.product_ind_not_stab))

        self.primary_oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        self.primary_oxidant.transform_to_descards()

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        secondary_active = np.array([np.sum(self.secondary_active.c3d[:, :, plane_ind]) for plane_ind
                                     in self.product_indexes], dtype=np.uint32)
        secondary_active_moles = secondary_active * Config.ACTIVES.SECONDARY.MOLES_PER_CELL
        secondary_outward_eq_mat_moles = secondary_active * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        secondary_product = np.array([np.sum(self.secondary_product.c3d[:, :, plane_ind]) for plane_ind
                                      in self.product_indexes], dtype=np.uint32)
        secondary_product_moles = secondary_product * Config.PRODUCTS.SECONDARY.MOLES_PER_CELL
        secondary_product_eq_mat_moles = secondary_product * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles - \
                       secondary_outward_eq_mat_moles - secondary_product_eq_mat_moles
        less_than_zero = np.where(matrix_moles < 0)[0]
        matrix_moles[less_than_zero] = 0
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles + \
                      secondary_active_moles + secondary_product_moles

        product_c = product_moles / whole_moles
        secondary_product_c = secondary_product_moles / whole_moles

        temp_ind_p = np.where(product_c == 0)[0]
        self.cases.first.prod_indexes[self.product_indexes[temp_ind_p]] = False
        product_c = np.delete(product_c, temp_ind_p)
        self.comb_indexes.append(np.delete(self.product_indexes, temp_ind_p))

        temp_ind_s = np.where(secondary_product_c == 0)[0]
        self.cases.second.prod_indexes[self.product_indexes[temp_ind_s]] = False
        secondary_product_c = np.delete(secondary_product_c, temp_ind_s)
        self.comb_indexes.append(np.delete(self.product_indexes, temp_ind_s))

        temp_ind = np.where(product_c > Config.PRODUCTS.PRIMARY.PHASE_FRACTION_LIMIT)[0]
        self.cases.first.product_ind_not_stab[self.comb_indexes[0][temp_ind]] = False
        self.comb_indexes[0] = np.delete(self.comb_indexes[0], temp_ind)

        temp_ind = np.where(secondary_product_c > Config.PRODUCTS.SECONDARY.PHASE_FRACTION_LIMIT)[0]
        self.cases.second.product_ind_not_stab[self.comb_indexes[1][temp_ind]] = False
        self.comb_indexes[1] = np.delete(self.comb_indexes[1], temp_ind)

        saved_ind = self.comb_indexes[1]
        if len(self.comb_indexes[0]) > 0:
            self.comb_indexes = self.comb_indexes[0]
            self.cur_case = self.cases.first
            self.cur_case_mp = self.cases.first_mp
            self.decomposition_intrinsic()

        if len(saved_ind) > 0:
            self.comb_indexes = saved_ind
            self.cur_case = self.cases.second
            self.cur_case_mp = self.cases.second_mp
            self.decomposition_intrinsic()

    def dissolution_atomic_stop_if_stable_MP(self):
        self.product_indexes = np.where(self.product_x_nzs)[0]

        where_not_stab = np.where(self.product_x_not_stab)[0]
        self.product_indexes = np.intersect1d(self.product_indexes, where_not_stab)

        new_stab_count = np.count_nonzero(~self.product_x_not_stab)
        if new_stab_count > self.prev_stab_count:
            self.prev_stab_count = new_stab_count
            print("stable now at: ", np.nonzero(~self.product_x_not_stab)[0])

        self.primary_oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        self.primary_oxidant.transform_to_descards()

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        less_than_zero = np.where(matrix_moles < 0)[0]
        matrix_moles[less_than_zero] = 0

        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        temp_ind = np.where(product == 0)[0]
        self.product_x_nzs[self.product_indexes[temp_ind]] = False

        product_c = np.delete(product_c, temp_ind)
        self.comb_indexes = np.delete(self.product_indexes, temp_ind)

        temp_ind = np.where(product_c > Config.PHASE_FRACTION_LIMIT)[0]
        self.product_x_not_stab[self.comb_indexes[temp_ind]] = False

        product_c = np.delete(product_c, temp_ind)
        self.comb_indexes = np.delete(self.comb_indexes, temp_ind)

        powers = self.powers[self.comb_indexes]
        soll_prod = Config.PROD_INCR_CONST * (self.curr_time - self.active_times[self.comb_indexes]) ** powers
        # soll_prod = Config.PROD_INCR_CONST * (self.curr_time - self.active_times[self.comb_indexes]) ** 1.1

        self.diffs = product_c - soll_prod

        temp = np.where(self.diffs > 0)[0]
        self.comb_indexes = self.comb_indexes[temp]

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()

    def dissolution_atomic_if_stable_higer_p(self):
        self.comb_indexes = np.where(self.product_x_nzs)[0]
        # where_not_stab = np.where(self.product_x_not_stab)[0]
        # self.product_indexes = np.intersect1d(self.product_indexes, where_not_stab)

        self.primary_oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.comb_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        self.primary_oxidant.transform_to_descards()

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.comb_indexes], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.comb_indexes], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        # temp_ind = np.where(product == 0)[0]
        # self.product_x_nzs[self.product_indexes[temp_ind]] = False

        # product_c = np.delete(product_c, temp_ind)
        # self.comb_indexes = np.delete(self.product_indexes, temp_ind)

        temp_ind = np.where(product_c > Config.PHASE_FRACTION_LIMIT)[0]
        # self.product_x_not_stab[self.comb_indexes[temp_ind]] = False
        # self.comb_indexes = np.delete(self.comb_indexes, temp_ind)
        frac = np.zeros(len(self.comb_indexes))
        frac[temp_ind] = 1

        if len(self.comb_indexes) > 0:
            self.cur_case.dissolution_probabilities.adapt_probabilities(self.comb_indexes, frac)
            self.decomposition_intrinsic()

    def dissolution_stop_if_no_active(self):
        # self.ioz_bound = self.get_cur_ioz_bound()
        self.comb_indexes = np.where(self.cur_case.prod_indexes)[0]

        # temp = np.where(self.comb_indexes <= self.ioz_bound)[0]
        # self.comb_indexes = self.comb_indexes[temp]

        active = np.array([np.sum(self.cur_case.active.c3d[:, :, plane_ind]) for plane_ind in self.comb_indexes], dtype=np.uint32)

        temp = np.where(active > 0)[0]
        self.comb_indexes = self.comb_indexes[temp]

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()

    def dissolution_atomic_stop_if_no_active_or_no_oxidant(self):
        self.product_indexes = np.where(self.product_x_nzs)[0]

        self.primary_oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        self.primary_oxidant.transform_to_descards()

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)

        temp_ind = np.where(product == 0)[0]
        self.product_x_nzs[self.product_indexes[temp_ind]] = False

        active = np.delete(active, temp_ind)
        oxidant = np.delete(oxidant, temp_ind)
        self.comb_indexes = np.delete(self.product_indexes, temp_ind)

        temp_ind = np.where(active == 0)[0]
        temp_ind1 = np.where(oxidant == 0)[0]

        temp_ind = np.unique(np.concatenate((temp_ind, temp_ind1)))

        self.comb_indexes = np.delete(self.comb_indexes, temp_ind)

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()

    def dissolution_atomic_with_kinetic(self):
        self.product_indexes = np.where(self.product_x_nzs)[0]

        self.primary_oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        self.primary_oxidant.transform_to_descards()

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        temp = np.where((product_c > Config.PHASE_FRACTION_LIMIT) | (product_c > self.soll_prod))[0]
        self.comb_indexes = self.product_indexes[temp]

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()

    def dissolution_atomic_with_kinetic_and_KP(self):
        self.product_indexes = np.where(self.product_x_nzs)[0]

        self.primary_oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        self.primary_oxidant.transform_to_descards()

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        soll_prod = Config.PROD_INCR_CONST * (self.curr_time - self.active_times[self.product_indexes]) ** 1.1
        self.diffs = product_c - soll_prod

        temp = np.where((product_c > Config.PHASE_FRACTION_LIMIT) | (self.diffs > 0))[0]
        self.comb_indexes = self.product_indexes[temp]

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()

    def dissolution_atomic_with_kinetic_lut_nicr5(self):
        self.product_indexes = np.where(self.cur_case.prod_indexes)[0]

        self.cur_case.oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.cur_case.oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        self.cur_case.oxidant.transform_to_descards()

        active = np.array([np.sum(self.cur_case.active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = self._get_product_counts_at_indexes_for_case(self.cur_case, self.cur_case_mp, self.product_indexes)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL_TC
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL * \
                               Config.PRODUCTS.PRIMARY.THRESHOLD_OUTWARD

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        temp_ind = np.where(product == 0)[0]
        self.cur_case.prod_indexes[self.product_indexes[temp_ind]] = False
        product_c = np.delete(product_c, temp_ind)
        self.comb_indexes = np.delete(self.product_indexes, temp_ind)

        times = np.full(len(self.comb_indexes), self.iteration * Config.GENERATED_VALUES.TAU)
        pos = self.comb_indexes * Config.GENERATED_VALUES.LAMBDA
        # print(times)
        # print(pos)
        curr_look_up = self.KinDATA.get_look_up_data(times, pos)

        # some = (curr_look_up * (1 + Config.PROD_ERROR))

        self.product_indexes = np.where(product_c > (curr_look_up * (1 + Config.PROD_ERROR)))[0]
        tind = np.where(product_c > curr_look_up)[0]

        self.comb_indexes = self.comb_indexes[tind]

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()

    def dissolution_atomic_with_kinetic_MP(self):
        self.product_indexes = np.where(self.product_x_nzs)[0]

        self.primary_oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        self.primary_oxidant.transform_to_descards()

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        soll_prod = Config.PROD_INCR_CONST * (self.curr_time - self.active_times[self.product_indexes]) ** 1.1

        # if self.iteration % Config.STRIDE == 0:
        #     itera = np.full(len(self.product_indexes), self.iteration) // Config.STRIDE
        #     indexes = np.stack((self.product_indexes, itera))
        #
        #     self.cumul_prod.set_at_ind(indexes, product_c)
        #     self.growth_rate.set_at_ind(indexes, soll_prod)
        self.diffs = product_c - soll_prod

        temp = np.where((product_c > Config.PHASE_FRACTION_LIMIT) | (self.diffs > 0))[0]
        self.comb_indexes = self.product_indexes[temp]

        if len(self.comb_indexes) > 0:
            dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                       Config.PRODUCTS.PRIMARY)

            tasks = [(self.primary_product.shm_mdata, chunk_range, self.comb_indexes, self.aggregated_ind,
                      dissolution_probabilities, dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL) for chunk_range in
                     self.chunk_ranges]

            results = list(self.pool.imap_unordered(worker, tasks))

            to_dissolve = np.array(np.concatenate(results, axis=1), dtype=np.ushort)
            if len(to_dissolve[0]) > 0:
                just_decrease_counts(self.primary_product.c3d, to_dissolve)
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                insert_counts(self.primary_active.c3d, to_dissolve)
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def dissolution_standard(self):
        # self.comb_indexes = self.get_cur_dissol_ioz_bound()
        self.product_indexes = np.where(self.cur_case.prod_indexes)[0]
        # temp = np.where(self.comb_indexes <= self.ioz_bound)[0]
        # self.comb_indexes = self.comb_indexes[temp]
        # not_stable_ind = np.where(self.product_x_not_stab)[0]
        # nz_ind = np.where(self.product_x_nzs)[0]
        # self.product_indexes = np.intersect1d(not_stable_ind, nz_ind)
        product = np.array([np.any(self.cur_case.product.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes])
        where_no_prod = np.where(~product)[0]
        self.cur_case.prod_indexes[self.product_indexes[where_no_prod]] = False
        self.comb_indexes = np.where(self.cur_case.prod_indexes)[0]

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()

    def ci_single_only_p1(self, seeds):
        all_arounds = self.utils.calc_sur_ind_formation(seeds, self.cur_case.active.c3d.shape[2] - 1)
        neighbours = go_around_bool(self.cur_case.active.c3d, all_arounds)
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)
        temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]

        # activate for dependent growth___________________________________________________________________
        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]
            flat_arounds = all_arounds[:, 0:self.cur_case.product.lind_flat_arr]
            # arr_len_in_flat = self.go_around(self.precipitations3d_init, flat_arounds)
            arr_len_in_flat = self.cur_case.go_around_func_ref(flat_arounds)
            homogeneous_ind = np.where(arr_len_in_flat == 0)[0]
            needed_prob = np.full(len(arr_len_in_flat), Config.PROBABILITIES.PRIMARY.p1)
            needed_prob[homogeneous_ind] = 0
            randomise = np.array(np.random.random_sample(arr_len_in_flat.size), dtype=np.float64)
            temp_ind = np.where(randomise < needed_prob)[0]
        # _________________________________________________________________________________________________

            if len(temp_ind) > 0:
                seeds = seeds[temp_ind]
                neighbours = neighbours[temp_ind]
                all_arounds = all_arounds[temp_ind]
                out_to_del = np.array(np.nonzero(neighbours))
                start_seed_index = np.unique(out_to_del[0], return_index=True)[1]
                to_del = np.array([out_to_del[1, indx:indx + self.threshold_outward] for indx in start_seed_index],
                                  dtype=np.ubyte)
                coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                                 dtype=np.short)
                coord = np.reshape(coord, (len(coord) * self.threshold_outward, 3))

                # exists = check_at_coord(self.objs[self.case]["product"].full_c3d, coord)  # precip on place of active!
                # exists = check_at_coord(self.objs[self.case]["product"].full_c3d, seeds)  # precip on place of oxidant!

                # temp_ind = np.where(exists)[0]
                # coord = np.delete(coord, temp_ind, 0)
                # seeds = np.delete(seeds, temp_ind, 0)

                # if self.objs[self.case]["to_check_with"] is not None:
                #     # to_check_min_self = np.array(self.cumul_product - product.c3d, dtype=np.ubyte)
                #     exists = np.array([self.objs[self.case]["to_check_with"].c3d[point[0], point[1], point[2]]
                #                        for point in coord], dtype=np.ubyte)
                #     # exists = np.array([to_check_min_self[point[0], point[1], point[2]] for point in coord],
                #     #                   dtype=np.ubyte)
                #     temp_ind = np.where(exists > 0)[0]
                #     coord = np.delete(coord, temp_ind, 0)
                #     seeds = np.delete(seeds, temp_ind, 0)

                coord = coord.transpose()
                seeds = seeds.transpose()

                self.cur_case.active.c3d[coord[0], coord[1], coord[2]] -= 1
                self.cur_case.oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1

                # self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
                self.cur_case.product.c3d[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

                # self.objs[self.case]["product"].fix_full_cells(coord)  # precip on place of active!
                self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!

                # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
                # dissolution function
                self.product_x_nzs[seeds[2][0]] = True

                # self.cumul_product[coord[0], coord[1], coord[2]] += 1

    def ci_single_no_growth_solub_prod_test(self, seeds):
        """
        Created only for tests of the solubility product probability function
        """
        all_arounds = self.utils.calc_sur_ind_formation(seeds, self.cur_case.active.c3d.shape[2] - 1)
        neighbours = go_around_bool(self.cur_case.active.c3d, all_arounds)
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)
        temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]

        # activate for dependent growth___________________________________________________________________
        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]
            # flat_arounds = all_arounds[:, 0:self.objs[self.case]["product"].lind_flat_arr]

            # flat_neighbours = self.go_around(self.precipitations3d_init_full, flat_arounds)
            # flat_neighbours = self.go_around(flat_arounds)
            # arr_len_in_flat = np.array([np.sum(item) for item in flat_neighbours], dtype=int)

            # arr_len_in_flat = self.go_around(self.precipitations3d_init, flat_arounds)

            # arr_len_in_flat = np.zeros(len(flat_arounds))  # REMOVE!!!!!!!!!!!!!!!!!!

            # homogeneous_ind = np.where(arr_len_in_flat == 0)[0]

            # needed_prob = self.nucl_prob.get_probabilities(arr_len_in_flat, seeds[0][2])
            needed_prob = self.nucl_prob.nucl_prob.values_pp[seeds[0][2]]  # seeds[0][2] - current plane index
            randomise = np.array(np.random.random_sample(len(seeds)), dtype=np.float64)
            temp_ind = np.where(randomise < needed_prob)[0]
            # _________________________________________________________________________________________________

            if len(temp_ind) > 0:
                seeds = seeds[temp_ind]
                neighbours = neighbours[temp_ind]
                all_arounds = all_arounds[temp_ind]
                out_to_del = np.array(np.nonzero(neighbours))
                start_seed_index = np.unique(out_to_del[0], return_index=True)[1]
                to_del = np.array([out_to_del[1, indx:indx + self.threshold_outward] for indx in start_seed_index],
                                  dtype=np.ubyte)
                coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                                 dtype=np.short)
                coord = np.reshape(coord, (len(coord) * self.threshold_outward, 3))

                coord = coord.transpose()
                seeds = seeds.transpose()

                self.cur_case.active.c3d[coord[0], coord[1], coord[2]] -= 1
                self.cur_case.oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1

                # self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
                self.cur_case.product.c3d[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

                # self.objs[self.case]["product"].fix_full_cells(coord)  # precip on place of active!
                self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!

    def ci_single_two_products_no_growth(self, seeds):
        all_arounds = self.utils.calc_sur_ind_formation(seeds, self.cur_case.active.c3d.shape[2] - 1)
        neighbours = go_around_bool(self.cur_case.active.c3d, all_arounds)
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)
        temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]

        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]
            out_to_del = np.array(np.nonzero(neighbours))
            start_seed_index = np.unique(out_to_del[0], return_index=True)[1]
            to_del = np.array([out_to_del[1, indx:indx + self.threshold_outward] for indx in start_seed_index],
                              dtype=np.ubyte)
            coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                             dtype=np.short)
            coord = np.reshape(coord, (len(coord) * self.threshold_outward, 3))

            # exists = check_at_coord(self.objs[self.case]["product"].full_c3d, coord)  # precip on place of active!
            # exists = check_at_coord(self.objs[self.case]["product"].full_c3d, seeds)  # precip on place of oxidant!

            # temp_ind = np.where(exists)[0]
            # coord = np.delete(coord, temp_ind, 0)
            # seeds = np.delete(seeds, temp_ind, 0)

            # if self.objs[self.case]["to_check_with"] is not None:
            #     # to_check_min_self = np.array(self.cumul_product - product.c3d, dtype=np.ubyte)
            #     exists = np.array([self.objs[self.case]["to_check_with"].c3d[point[0], point[1], point[2]]
            #                        for point in coord], dtype=np.ubyte)
            #     # exists = np.array([to_check_min_self[point[0], point[1], point[2]] for point in coord],
            #     #                   dtype=np.ubyte)
            #     temp_ind = np.where(exists > 0)[0]
            #     coord = np.delete(coord, temp_ind, 0)
            #     seeds = np.delete(seeds, temp_ind, 0)

            coord = coord.transpose()
            seeds = seeds.transpose()

            self.cur_case.active.c3d[coord[0], coord[1], coord[2]] -= 1
            self.cur_case.oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1

            # self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
            self.cur_case.product.c3d[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

            # self.cur_case.product.fix_full_cells(coord)  # precip on place of active!
            self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!

    def diffuse_all(self):
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

    def calc_precipitation_front_only_cells(self):
        """
        Calculating a position of a precipitation front, considering only cells concentrations without any scaling!
        As a boundary a product fraction of 0,1% is used.
        """
        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.cells_per_axis)], dtype=np.uint32)
        product = product / (self.cells_per_axis ** 2)
        threshold = Config.ACTIVES.PRIMARY.CELLS_CONCENTRATION
        for rev_index, precip_conc in enumerate(np.flip(product)):
            if precip_conc > threshold / 100:
                position = (len(product) - 1 - rev_index) * Config.SIZE * 10 ** 6 \
                           / self.cells_per_axis
                sqr_time = ((self.iteration + 1) * Config.SIM_TIME / (self.n_iter * 3600)) ** (1 / 2)
                self.utils.db.insert_precipitation_front(sqr_time, position, "p")
                break

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
        z_hi = u_bound + 2
        self.cases.precip_3d_init[:, :, 0:z_hi] = 0
        self.cases.precip_3d_init[:, :, 0:z_hi] = self.cases.product_state[1, :, :, 0:z_hi]

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

    def go_around_single_oxid_n(self, around_coords):
        return np.sum(go_around_bool(self.cur_case.precip_3d_init, around_coords), axis=1)

    def go_around_mult_oxid_n(self, around_coords):
        all_neigh = go_around_int(self.cur_case.precip_3d_init, around_coords)
        neigh_in_prod = all_neigh[:, 6].view()
        nonzero_neigh_in_prod = np.array(np.nonzero(neigh_in_prod)[0])
        where_full_side_neigh = np.unique(np.where(all_neigh[:, :6].view() == self.cur_case.product.oxidation_number)[0])
        only_inside_product = np.setdiff1d(nonzero_neigh_in_prod, where_full_side_neigh, assume_unique=True)
        final_effective_flat_counts = np.zeros(len(all_neigh), dtype=np.ubyte)
        final_effective_flat_counts[where_full_side_neigh] = np.sum(all_neigh[where_full_side_neigh], axis=1)
        final_effective_flat_counts[only_inside_product] = 7 * self.cur_case.product.oxidation_number - 1
        return final_effective_flat_counts

    def go_around_mult_oxid_n_also_partial_neigh(self, around_coords):
        """Im Gegensatz zu go_around_mult_oxid_n werden auch die parziellen Nachbarn (weniger als oxidation numb inside)
        berücksichtigt!
        Resolution inside a product: If inside a product the probability is equal to ONE!!"""
        all_neigh = go_around_int(self.cur_case.precip_3d_init, around_coords)
        neigh_in_prod = all_neigh[:, 6].view()
        nonzero_neigh_in_prod = np.array(np.nonzero(neigh_in_prod)[0])
        final_effective_flat_counts = np.sum(all_neigh, axis=1)
        final_effective_flat_counts[nonzero_neigh_in_prod] = 7 * self.cur_case.product.oxidation_number - 1
        return final_effective_flat_counts

    def go_around_mult_oxid_n_also_partial_neigh_aip(self, around_coords):
        """Im Gegensatz zu go_around_mult_oxid_n werden auch die parziellen Nachbarn (weniger als oxidation numb inside)
        berücksichtigt!!!
        aip: Adjusted Inside Product!
        Resolution inside a product: probability adjusted according to a number of neighbours"""
        return np.sum(go_around_int(self.cur_case.precip_3d_init, around_coords), axis=1)

    def go_around_single_oxid_n_single_neigh(self, around_coords):
        """Does not distinguish between multiple flat neighbours. If at least one flat neighbour P=P1"""
        flat_neighbours = go_around_bool(self.cur_case.precip_3d_init, around_coords)
        temp = np.array([np.sum(item) for item in flat_neighbours], dtype=bool)

        return np.array(temp, dtype=np.ubyte)

    def go_around_mult_oxid_n_single_neigh(self, around_coords):
        """Does not distinguish between multiple flat neighbours. If at least one flat neighbour P=P1"""

        all_neigh = go_around_int(self.cur_case.precip_3d_init, around_coords)
        neigh_in_prod = all_neigh[:, 6].view()
        nonzero_neigh_in_prod = np.array(np.nonzero(neigh_in_prod)[0])
        where_full_side_neigh = np.unique(np.where(all_neigh[:, :6].view() == self.cur_case.product.oxidation_number)[0])
        only_inside_product = np.setdiff1d(nonzero_neigh_in_prod, where_full_side_neigh, assume_unique=True)
        final_effective_flat_counts = np.zeros(len(all_neigh), dtype=np.ubyte)
        final_effective_flat_counts[where_full_side_neigh] = self.cur_case.product.oxidation_number
        final_effective_flat_counts[only_inside_product] = 7 * self.cur_case.product.oxidation_number - 1
        return final_effective_flat_counts


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

    def dissolution_mp_subblock(self):
        """
        Dissolution V2: product snapshot for consistent neighbour reads; workers write directly
        to oxidant write buffer (count + dirs grid) and active. No flat cells/dirs; oxidant is
        in shared memory. Optional block logic via aggregated_ind and bsf.
        """
        dp = self.cur_case.dissolution_probabilities
        if dp is None:
            return
        comb = np.asarray(self.comb_indexes, dtype=np.intp).ravel()
        if comb.size == 0:
            return
        oxidant_elem = self.cur_case.oxidant
        if not hasattr(oxidant_elem, "get_current_c3d_shm_mdata"):
            return
        n_workers = max(1, getattr(self.worker_pools, "n_outward_workers", getattr(self.diffusion_engine, "n_outward_workers", 4)))
        n_z = self.cur_case_mp.product_c3d_shm_mdata.shape[2]
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

        # Product snapshot (read-only for workers)
        shm_product = shared_memory.SharedMemory(name=self.cur_case_mp.product_c3d_shm_mdata.name)
        product = np.ndarray(
            self.cur_case_mp.product_c3d_shm_mdata.shape,
            dtype=self.cur_case_mp.product_c3d_shm_mdata.dtype,
            buffer=shm_product.buf,
        )
        snapshot_shm = shared_memory.SharedMemory(create=True, size=product.nbytes)
        snapshot = np.ndarray(product.shape, dtype=product.dtype, buffer=snapshot_shm.buf)
        np.copyto(snapshot, product)
        shm_product.close()
        product_snapshot_mdata = SharedMetaData(snapshot_shm.name, snapshot.shape, snapshot.dtype)

        # Add new oxidant to current (read) buffer so particles are in the active state and survive next diffusion
        oxidant_write_mdata = oxidant_elem.get_current_c3d_shm_mdata()
        max_per_cell_oxidant = oxidant_elem.max_per_cell
        active_elem = self.cur_case.active
        max_per_cell_active = active_elem.max_per_cell
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

        pool = self.worker_pools.dissolution_pool
        pool.map(dissolution_subblock_worker, tasks)

        snapshot_shm.close()
        try:
            snapshot_shm.unlink()
        except FileNotFoundError:
            pass


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
        # Point case_mp at current read buffers (diffusion may have swapped A/B)
        # self.cur_case = self.cases.product_cases[0]
        # self.get_combi_ind()
        cur_case_mp.oxidant_c3d_shm_mdata = cur_case.oxidant.get_current_c3d_shm_mdata()
        cur_case_mp.active_c3d_shm_mdata = cur_case.active.get_current_c3d_shm_mdata()
        cur_case.fix_init_precip_func_ref(self.cells_per_axis)

        # Two-state strategy only: even iterations -> base, odd iterations -> mirrored.
        if int(self.iteration) % 2 == 0:
            z_blocks = self._precip_z_blocks
            gap_z_groups = self._precip_gap_z_groups
        else:
            z_blocks = self._precip_z_blocks_neg
            gap_z_groups = self._precip_gap_z_groups_neg

        plane_indexes = np.asarray(cur_case_mp.plane_indexes, dtype=np.intp)
        ind_form = np.asarray(ind_formation, dtype=np.int8)

        # Interior z-blocks:
        tasks_std = []
        tasks_tail_seq = []
        for segs in z_blocks:
            if len(segs) == 1:
                k_lo, k_hi = segs[0]
                tasks_std.append((cur_case_mp, k_lo, k_hi, plane_indexes, cur_case.oxidant.max_per_cell, cur_case.active.max_per_cell, ind_form))
            else:
                # Use the largest segment in the parallel pass; defer the remaining
                # wrapped segment(s) to a short sequential tail pass.
                segs_sorted = sorted(segs, key=lambda ab: (ab[1] - ab[0] + 1), reverse=True)
                main_seg = segs_sorted[0]
                tail_segs = segs_sorted[1:]
                k_lo, k_hi = main_seg
                tasks_std.append((cur_case_mp, k_lo, k_hi, plane_indexes, cur_case.oxidant.max_per_cell, cur_case.active.max_per_cell, ind_form))
                for k_lo, k_hi in tail_segs:
                    tasks_tail_seq.append((cur_case_mp, k_lo, k_hi, plane_indexes, cur_case.oxidant.max_per_cell, cur_case.active.max_per_cell, ind_form))
        if tasks_std:
            self.worker_pools.nucleation_pool.map(precip_step_subblock_worker, tasks_std)
        for task in tasks_tail_seq:
            precip_step_subblock_worker(task)
        # Gap z-planes: run all gap indices in one worker task using explicit seed list.
        gap_seed_slab_k = list(dict.fromkeys(int(k) for group in gap_z_groups for k in group))
        gap_task = [(
            cur_case_mp,
            0,
            int(cur_case_mp.oxidant_c3d_shm_mdata.shape[2]) - 1,
            plane_indexes,
            cur_case.oxidant.max_per_cell,
            cur_case.active.max_per_cell,
            ind_form,
            gap_seed_slab_k,
        )]
        self.worker_pools.nucleation_pool.map(precip_step_subblock_worker, gap_task)
    
    def nucleate(self):
        self.get_combi_ind()
        for case, case_mp in self.cases.product_case_pairs:
            if case_mp.plane_indexes:
                self.precip_mp_subblock(case, case_mp)

    def ioz_depth_from_kinetics(self):
        self.curr_time = Config.GENERATED_VALUES.TAU * (self.iteration + 1)
        active_ind = np.where(self.active_times <= self.curr_time)[0]
        return min(np.amax(active_ind), self.furthest_index)

    def ioz_depth_furthest_inward(self):
        oxidant_3d = self.cur_case.oxidant.get_3d_grid()[0]
        # ioz_bound = max x (i) where any oxidant or active particle exists (narrow the domain)
        flat_o = np.flatnonzero(oxidant_3d.ravel(order="F") > 0)
        max_x_ox = int(np.max(flat_o % self.cells_per_axis)) if flat_o.size > 0 else -1
        self.ioz_bound = max(max_x_ox, 0)
        return self.ioz_bound

    def ioz_dissolution_where_prod(self):
        return np.where(self.cur_case.prod_indexes)[0]
