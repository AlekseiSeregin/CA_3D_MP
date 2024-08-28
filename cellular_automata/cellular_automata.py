import sys
import utils
import elements
import multiprocessing
from .nes_for_mp import *
from .nucleation_functions import *
from .dissolution_functions import *
from thermodynamics import td_data


class CellularAutomata:
    def __init__(self):
        self.utils = utils.Utils()
        self.utils.generate_param()
        self.cases = utils.CaseRef()
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
        # inward
        self.primary_oxidant = None  # in the future must be hosted in the case ref!
        self.secondary_oxidant = None
        # outward
        self.primary_active = None
        self.secondary_active = None
        # functions
        self.precip_func = None  # must be defined elsewhere
        self.get_combi_ind = None  # must be defined elsewhere
        self.precip_step = None  # must be defined elsewhere
        self.get_cur_ioz_bound = None  # must be defined elsewhere
        self.get_cur_dissol_ioz_bound = None  # must be defined elsewhere
        self.check_intersection = None  # must be defined elsewhere
        self.decomposition = None  # must be defined elsewhere
        self.decomposition_intrinsic = None  # must be defined elsewhere

        self.coord_buffer = None
        self.to_dissol_pn_buffer = None

        self.primary_product = None
        self.primary_oxid_numb = Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER
        self.max_inside_neigh_number = 6 * self.primary_oxid_numb
        self.max_block_neigh_number = 7

        self.disol_block_p = Config.PROBABILITIES.PRIMARY.p0_d ** Config.PROBABILITIES.PRIMARY.n
        self.disol_p = Config.PROBABILITIES.PRIMARY.p0_d

        self.secondary_product = None
        self.ternary_product = None
        self.quaternary_product = None
        self.quint_product = None

        self.primary_fetch_ind = []
        self.secondary_fetch_ind = []
        self.fetch_ind = None
        self.generate_fetch_ind_mp()

        self.aggregated_ind = np.array([[7, 0, 1, 2, 19, 16, 14],
                                        [6, 0, 1, 5, 18, 15, 14],
                                        [8, 0, 4, 5, 20, 15, 17],
                                        [9, 0, 4, 2, 21, 16, 17],
                                        [11, 3, 1, 2, 19, 24, 22],
                                        [10, 3, 1, 5, 18, 23, 22],
                                        [12, 3, 4, 5, 20, 23, 25],
                                        [13, 3, 4, 2, 21, 24, 25]], dtype=np.int64)

        self.numb_of_proc = Config.NUMBER_OF_PROCESSES
        if self.cells_per_axis % self.numb_of_proc == 0:
            chunk_size = int(self.cells_per_axis / self.numb_of_proc)
        else:
            chunk_size = int((self.cells_per_axis - 1) // (self.numb_of_proc - 1))

        self.chunk_ranges = np.zeros((self.numb_of_proc, 2), dtype=int)
        self.chunk_ranges[0] = [0, chunk_size]

        for pos in range(1, self.numb_of_proc):
            self.chunk_ranges[pos, 0] = self.chunk_ranges[pos-1, 1]
            self.chunk_ranges[pos, 1] = self.chunk_ranges[pos, 0] + chunk_size
        self.chunk_ranges[-1, 1] = self.cells_per_axis

        self.pool = multiprocessing.Pool(processes=self.numb_of_proc, maxtasksperchild=Config.MAX_TASK_PER_CHILD)

        self.threshold_inward = Config.THRESHOLD_INWARD
        self.threshold_outward = Config.THRESHOLD_OUTWARD

        self.comb_indexes = None
        self.rel_prod_fraction = None
        self.gamma_primes = None
        self.product_indexes = None
        self.nucleation_indexes = None

        self.save_flag = False

        self.product_x_not_stab = np.full(self.cells_per_axis, True, dtype=bool)
        self.TdDATA = td_data.TdDATA()
        self.TdDATA.fetch_look_up_from_file()
        self.curr_look_up = None

        self.prev_stab_count = 0

        self.precipitation_stride = Config.STRIDE * Config.STRIDE_MULTIPLIER

        self.save_rate = self.n_iter // Config.STRIDE
        self.cumul_prod = utils.my_data_structs.MyBufferSingle((self.cells_per_axis, self.save_rate), dtype=float)
        self.growth_rate = utils.my_data_structs.MyBufferSingle((self.cells_per_axis, self.save_rate), dtype=float)

        self.diffs = None
        self.curr_time = 0

        lambdas = (np.arange(self.cells_per_axis, dtype=int) + 0.5) * Config.GENERATED_VALUES.LAMBDA
        adj_lamd = lambdas - Config.ZETTA_ZERO
        neg_ind = np.where(adj_lamd < 0)[0]
        adj_lamd[neg_ind] = 0
        self.active_times = adj_lamd ** 2 / Config.GENERATED_VALUES.KINETIC_KONST ** 2

        self.prev_len = 0
        self.powers = utils.physical_data.POWERS

    def dissolution_zhou_wei_original(self):
        """Implementation of original not adapted Zhou and Wei approach. Only two probabilities p for block and pn
        are considered. Works for any oxidation nuber!"""

        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, self.product_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.product_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            all_arounds = self.utils.calc_sur_ind_decompose(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, all_arounds)

            # all_neigh_pn = all_neigh[[]]
            # all_neigh_block = all_neigh[[]]

            # choose all the coordinates which have at least one full side neighbour
            where_full = np.unique(np.where(all_neigh[:, :6].view() == self.primary_oxid_numb)[0])

            to_dissol_pn_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_full), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_full))
            # self.to_dissol_pn_buffer.append_to_buffer(self.coord_buffer.get_elem_instead_ind(where_full))

            if self.coord_buffer.last_in_buffer > 0:
                all_neigh = all_neigh[where_full]

                arr_len_flat = np.array([np.sum(item[:6]) for item in all_neigh], dtype=np.ubyte)
                index_outside = np.where((arr_len_flat < self.max_inside_neigh_number))[0]

                self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
                all_neigh = all_neigh[index_outside]

                aggregation = np.array([[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh],
                                       dtype=np.ubyte)
                ind_where_blocks = np.unique(np.where(aggregation == self.max_block_neigh_number)[0])

                if len(ind_where_blocks) > 0:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_instead_ind(ind_where_blocks))
                    # all_neigh_pn = np.delete(all_neigh, ind_where_blocks, axis=0)

                    self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(ind_where_blocks))
                    # all_neigh_block = all_neigh[ind_where_blocks]
                else:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_buffer())
                    # all_neigh_pn = all_neigh

                    self.coord_buffer.reset_buffer()
                    # all_neigh_block = all_neigh[[]]

            # probs_pn_no_neigh = self.dissol_prob.dissol_prob.values_pp[to_dissol_pn_no_neigh[2]]
            probs_pn_no_neigh = np.full(len(to_dissol_pn_no_neigh[0]), self.disol_p)

            to_dissolve_pn = self.to_dissol_pn_buffer.get_buffer()
            # all_neigh_pn = np.array([np.sum(item[:6]) for item in all_neigh_pn])
            # all_neigh_pn = np.zeros(len(all_neigh_pn))
            # probs_pn = self.dissol_prob.dissol_prob.values_pp[to_dissolve_pn[2]]
            probs_pn = np.full(len(to_dissolve_pn[0]), self.disol_p)

            to_dissolve_p = self.coord_buffer.get_buffer()
            # all_neigh_block = np.array([np.sum(item[:6]) for item in all_neigh_block])
            # all_neigh_block = np.full(len(all_neigh_block), self.primary_oxid_numb * 3)
            # probs_p = self.dissol_prob.get_probabilities_block(all_neigh_block, to_dissolve_p[2])
            probs_p = np.full(len(to_dissolve_p[0]), self.disol_block_p)

            randomise = np.random.random_sample(len(to_dissol_pn_no_neigh[0]))
            temp_ind = np.where(randomise < probs_pn_no_neigh)[0]
            to_dissol_pn_no_neigh = to_dissol_pn_no_neigh[:, temp_ind]

            randomise = np.random.random_sample(len(to_dissolve_pn[0]))
            temp_ind = np.where(randomise < probs_pn)[0]
            to_dissolve_pn = to_dissolve_pn[:, temp_ind]

            randomise = np.random.random_sample(len(to_dissolve_p[0]))
            temp_ind = np.where(randomise < probs_p)[0]
            to_dissolve_p = to_dissolve_p[:, temp_ind]

            to_dissolve = np.concatenate((to_dissolve_pn, to_dissolve_p, to_dissol_pn_no_neigh), axis=1)

            self.coord_buffer.reset_buffer()
            self.to_dissol_pn_buffer.reset_buffer()

            if len(to_dissolve[0]) > 0:
                counts = self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]]

                self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = 0
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                self.primary_active.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] += counts

                to_dissolve = np.repeat(to_dissolve, counts, axis=1)
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def dissolution_zhou_wei_with_bsf(self):
        """Implementation of Zhou and Wei approach. Works for any oxidation nuber!"""
        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, self.product_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.product_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            all_arounds = self.utils.calc_sur_ind_decompose(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, all_arounds)

            all_neigh_pn = all_neigh[[]]
            all_neigh_block = all_neigh[[]]

            # choose all the coordinates which have at least one full side neighbour
            where_full = np.unique(np.where(all_neigh[:, :6].view() == self.primary_oxid_numb)[0])

            to_dissol_pn_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_full), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_full))

            if self.coord_buffer.last_in_buffer > 0:
                all_neigh = all_neigh[where_full]

                arr_len_flat = np.array([np.sum(item[:6]) for item in all_neigh], dtype=np.ubyte)
                index_outside = np.where((arr_len_flat < self.max_inside_neigh_number))[0]

                self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
                all_neigh = all_neigh[index_outside]

                aggregation = np.array([[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh],
                                       dtype=np.ubyte)
                ind_where_blocks = np.unique(np.where(aggregation == self.max_block_neigh_number)[0])

                if len(ind_where_blocks) > 0:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_instead_ind(ind_where_blocks))
                    all_neigh_pn = np.delete(all_neigh, ind_where_blocks, axis=0)

                    self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(ind_where_blocks))
                    all_neigh_block = all_neigh[ind_where_blocks]
                else:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_buffer())
                    all_neigh_pn = all_neigh

                    self.coord_buffer.reset_buffer()
                    all_neigh_block = all_neigh[[]]

            probs_pn_no_neigh = self.dissol_prob.dissol_prob.values_pp[to_dissol_pn_no_neigh[2]]

            to_dissolve_pn = self.to_dissol_pn_buffer.get_buffer()
            all_neigh_pn = np.array([np.sum(item[:6]) for item in all_neigh_pn])
            probs_pn = self.dissol_prob.get_probabilities(all_neigh_pn, to_dissolve_pn[2])

            to_dissolve_p = self.coord_buffer.get_buffer()
            all_neigh_block = np.array([np.sum(item[:6]) for item in all_neigh_block])
            probs_p = self.dissol_prob.get_probabilities_block(all_neigh_block, to_dissolve_p[2])

            randomise = np.random.random_sample(len(to_dissol_pn_no_neigh[0]))
            temp_ind = np.where(randomise < probs_pn_no_neigh)[0]
            to_dissol_pn_no_neigh = to_dissol_pn_no_neigh[:, temp_ind]

            randomise = np.random.random_sample(len(to_dissolve_pn[0]))
            temp_ind = np.where(randomise < probs_pn)[0]
            to_dissolve_pn = to_dissolve_pn[:, temp_ind]

            randomise = np.random.random_sample(len(to_dissolve_p[0]))
            temp_ind = np.where(randomise < probs_p)[0]
            to_dissolve_p = to_dissolve_p[:, temp_ind]

            to_dissolve = np.concatenate((to_dissolve_pn, to_dissolve_p, to_dissol_pn_no_neigh), axis=1)

            self.coord_buffer.reset_buffer()
            self.to_dissol_pn_buffer.reset_buffer()

            if len(to_dissolve[0]) > 0:
                counts = self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]]

                self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = 0
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                self.primary_active.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] += counts

                to_dissolve = np.repeat(to_dissolve, counts, axis=1)
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def dissolution_zhou_wei_with_bsf_aip(self):
        """Implementation of Zhou and Wei approach. Works for any oxidation nuber!"""

        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, self.comb_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.comb_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            flat_arounds = self.utils.calc_sur_ind_decompose_flat_with_zero(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, flat_arounds)
            all_neigh[:, 6] -= 1

            all_neigh_block = np.array([])
            all_neigh_no_block = np.array([])
            numb_in_prod_block = np.array([], dtype=int)
            numb_in_prod_no_block = np.array([], dtype=int)

            where_not_null = np.unique(np.where(all_neigh > 0)[0])
            to_dissol_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_not_null), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_not_null))

            if self.coord_buffer.last_in_buffer > 0:
                all_neigh = all_neigh[where_not_null]
                arr_len_flat = np.sum(all_neigh[:, :6], axis=1)

                index_outside = np.where((arr_len_flat < self.max_inside_neigh_number))[0]
                self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
                all_neigh = all_neigh[index_outside]

                non_flat_arounds = self.utils.calc_sur_ind_decompose_no_flat(self.coord_buffer.get_buffer())
                non_flat_neigh = go_around_int(self.primary_product.c3d, non_flat_arounds)
                in_prod_column = np.array([all_neigh[:, 6]]).transpose()
                all_neigh = np.concatenate((all_neigh[:, :6], non_flat_neigh, in_prod_column), axis=1)
                numb_in_prod = all_neigh[:, -1]

                all_neigh_bool = np.array(all_neigh, dtype=bool)

                # aggregation = np.array([[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh_bool],
                #                        dtype=np.ubyte)
                # ind_where_blocks = np.unique(np.where(aggregation == 7)[0])

                ind_where_blocks = aggregate(self.aggregated_ind, all_neigh_bool)

                # if len(ind_where_blocks) > 0:
                #
                #     begin = time.time()
                #     aggregation = np.array(
                #         [[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh_bool],
                #         dtype=np.ubyte)
                #     ind_where_blocks = np.unique(np.where(aggregation == 7)[0])
                #     print("list comp: ", time.time() - begin)
                #
                #     begin = time.time()
                #     ind_where_blocks2 = aggregate(self.aggregated_ind, all_neigh_bool)
                #     print("numba: ", time.time() - begin)

                if len(ind_where_blocks) > 0:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_instead_ind(ind_where_blocks))
                    all_neigh_no_block = np.delete(all_neigh[:, :6], ind_where_blocks, axis=0)
                    numb_in_prod_no_block = np.delete(numb_in_prod, ind_where_blocks, axis=0)
                    # all_neigh_no_block = np.sum(all_neigh_no_block[:, :6], axis=1) + numb_in_prod_no_block
                    all_neigh_no_block = np.sum(all_neigh_no_block, axis=1) + numb_in_prod_no_block

                    self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(ind_where_blocks))
                    # all_neigh_block = all_neigh[ind_where_blocks]
                    all_neigh_block = all_neigh[ind_where_blocks, :6]
                    numb_in_prod_block = numb_in_prod[ind_where_blocks]
                    # all_neigh_block = np.sum(all_neigh_block[:, :6], axis=1) + numb_in_prod_block
                    all_neigh_block = np.sum(all_neigh_block, axis=1) + numb_in_prod_block
                else:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_buffer())
                    all_neigh_no_block = np.sum(all_neigh[:, :6], axis=1) + numb_in_prod
                    numb_in_prod_no_block = numb_in_prod

                    self.coord_buffer.reset_buffer()
                    all_neigh_block = np.array([])
                    numb_in_prod_block = np.array([], dtype=int)

            to_dissolve_no_block = self.to_dissol_pn_buffer.get_buffer()
            probs_no_block = self.cur_case.dissolution_probabilities.get_probabilities(all_neigh_no_block, to_dissolve_no_block[2])
            non_z_ind = np.where(numb_in_prod_no_block != 0)[0]
            repeated_coords = np.repeat(to_dissolve_no_block[:, non_z_ind], numb_in_prod_no_block[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_no_block[non_z_ind], numb_in_prod_no_block[non_z_ind])
            to_dissolve_no_block = np.concatenate((to_dissolve_no_block, repeated_coords), axis=1)
            probs_no_block = np.concatenate((probs_no_block, repeated_probs))
            randomise = np.random.random_sample(len(to_dissolve_no_block[0]))
            temp_ind = np.where(randomise < probs_no_block)[0]
            to_dissolve_no_block = to_dissolve_no_block[:, temp_ind]

            to_dissolve_block = self.coord_buffer.get_buffer()
            probs_block = self.cur_case.dissolution_probabilities.get_probabilities_block(all_neigh_block, to_dissolve_block[2])
            non_z_ind = np.where(numb_in_prod_block != 0)[0]
            repeated_coords = np.repeat(to_dissolve_block[:, non_z_ind], numb_in_prod_block[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_block[non_z_ind], numb_in_prod_block[non_z_ind])
            to_dissolve_block = np.concatenate((to_dissolve_block, repeated_coords), axis=1)
            probs_block = np.concatenate((probs_block, repeated_probs))
            randomise = np.random.random_sample(len(to_dissolve_block[0]))
            temp_ind = np.where(randomise < probs_block)[0]
            to_dissolve_block = to_dissolve_block[:, temp_ind]

            probs_no_neigh = self.cur_case.dissolution_probabilities.dissol_prob.values_pp[to_dissol_no_neigh[2]]
            randomise = np.random.random_sample(len(to_dissol_no_neigh[0]))
            temp_ind = np.where(randomise < probs_no_neigh)[0]
            to_dissol_no_neigh = to_dissol_no_neigh[:, temp_ind]

            to_dissolve = np.concatenate((to_dissolve_no_block, to_dissol_no_neigh, to_dissolve_block), axis=1)

            self.coord_buffer.reset_buffer()
            self.to_dissol_pn_buffer.reset_buffer()

            if len(to_dissolve[0]) > 0:
                just_decrease_counts(self.primary_product.c3d, to_dissolve)
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                insert_counts(self.primary_active.c3d, to_dissolve)
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def dissolution_zhou_wei_with_bsf_aip_UPGRADE(self):
        """Implementation of Zhou and Wei approach. Works for any oxidation nuber!
        Here the problem was that the geometrical arrangement is considered properly! For higher oxidation numbers the
         total number of neighbours does correlate with the geometrical configuration of the cluster!! 3 neighbours here
         mean to 3 geometrical flat neighbours."""

        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, self.comb_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.comb_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            flat_arounds = self.utils.calc_sur_ind_decompose_flat_with_zero(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, flat_arounds)
            all_neigh[:, 6] -= 1

            all_neigh_block = np.array([])
            all_neigh_no_block = np.array([])
            numb_in_prod_block = np.array([], dtype=int)
            numb_in_prod_no_block = np.array([], dtype=int)

            where_not_null = np.unique(np.where(all_neigh > 0)[0])
            to_dissol_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_not_null), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_not_null))

            if self.coord_buffer.last_in_buffer > 0:
                all_neigh = all_neigh[where_not_null]
                arr_len_flat = np.sum(all_neigh[:, :6], axis=1)

                index_outside = np.where((arr_len_flat < self.max_inside_neigh_number))[0]
                self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
                all_neigh = all_neigh[index_outside]

                non_flat_arounds = self.utils.calc_sur_ind_decompose_no_flat(self.coord_buffer.get_buffer())
                non_flat_neigh = go_around_int(self.primary_product.c3d, non_flat_arounds)
                in_prod_column = np.array([all_neigh[:, 6]]).transpose()
                all_neigh = np.concatenate((all_neigh[:, :6], non_flat_neigh, in_prod_column), axis=1)
                numb_in_prod = all_neigh[:, -1]

                all_neigh_bool = np.array(all_neigh, dtype=bool)

                # aggregation = np.array([[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh_bool],
                #                        dtype=np.ubyte)
                # ind_where_blocks = np.unique(np.where(aggregation == 7)[0])

                ind_where_blocks = aggregate(self.aggregated_ind, all_neigh_bool)
                # block_counts = aggregate_and_count(self.aggregated_ind, all_neigh_bool)
                # some = np.where(block_counts > 4)[0]
                #
                # if len(some) > 0:
                #     print()
                # ind_where_blocks = np.where(block_counts)[0]

                # if len(ind_where_blocks) > 0:
                #
                #     begin = time.time()
                #     aggregation = np.array(
                #         [[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh_bool],
                #         dtype=np.ubyte)
                #     ind_where_blocks = np.unique(np.where(aggregation == 7)[0])
                #     print("list comp: ", time.time() - begin)
                #
                #     begin = time.time()
                #     ind_where_blocks2 = aggregate(self.aggregated_ind, all_neigh_bool)
                #     print("numba: ", time.time() - begin)

                if len(ind_where_blocks) > 0:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_instead_ind(ind_where_blocks))
                    all_neigh_no_block = np.delete(all_neigh[:, :6], ind_where_blocks, axis=0)

                    all_neigh_bool = np.delete(all_neigh_bool[:, :6], ind_where_blocks, axis=0)
                    all_neigh_bool = np.sum(all_neigh_bool, axis=1)
                    ind_to_raise = np.where((all_neigh_bool == 3) | (all_neigh_bool == 4))[0]

                    numb_in_prod_no_block = np.delete(numb_in_prod, ind_where_blocks, axis=0)
                    # all_neigh_no_block = np.sum(all_neigh_no_block[:, :6], axis=1) + numb_in_prod_no_block
                    all_neigh_no_block = np.sum(all_neigh_no_block, axis=1) + numb_in_prod_no_block

                    all_neigh_no_block[ind_to_raise] = 0

                    self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(ind_where_blocks))
                    # all_neigh_block = all_neigh[ind_where_blocks]
                    all_neigh_block = all_neigh[ind_where_blocks, :6]
                    numb_in_prod_block = numb_in_prod[ind_where_blocks]
                    # all_neigh_block = np.sum(all_neigh_block[:, :6], axis=1) + numb_in_prod_block
                    all_neigh_block = np.sum(all_neigh_block, axis=1) + numb_in_prod_block
                else:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_buffer())
                    all_neigh_no_block = np.sum(all_neigh[:, :6], axis=1) + numb_in_prod

                    all_neigh_bool = np.sum(all_neigh_bool[:, :6], axis=1)
                    ind_to_raise = np.where((all_neigh_bool == 3) | (all_neigh_bool == 4))[0]

                    all_neigh_no_block[ind_to_raise] = 0

                    numb_in_prod_no_block = numb_in_prod

                    self.coord_buffer.reset_buffer()
                    all_neigh_block = np.array([])
                    numb_in_prod_block = np.array([], dtype=int)

            to_dissolve_no_block = self.to_dissol_pn_buffer.get_buffer()
            probs_no_block = self.cur_case.dissolution_probabilities.get_probabilities(all_neigh_no_block,
                                                                                       to_dissolve_no_block[2])
            non_z_ind = np.where(numb_in_prod_no_block != 0)[0]
            repeated_coords = np.repeat(to_dissolve_no_block[:, non_z_ind], numb_in_prod_no_block[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_no_block[non_z_ind], numb_in_prod_no_block[non_z_ind])
            to_dissolve_no_block = np.concatenate((to_dissolve_no_block, repeated_coords), axis=1)
            probs_no_block = np.concatenate((probs_no_block, repeated_probs))
            randomise = np.random.random_sample(len(to_dissolve_no_block[0]))
            temp_ind = np.where(randomise < probs_no_block)[0]
            to_dissolve_no_block = to_dissolve_no_block[:, temp_ind]

            to_dissolve_block = self.coord_buffer.get_buffer()
            probs_block = self.cur_case.dissolution_probabilities.get_probabilities_block(all_neigh_block,
                                                                                          to_dissolve_block[2])
            non_z_ind = np.where(numb_in_prod_block != 0)[0]
            repeated_coords = np.repeat(to_dissolve_block[:, non_z_ind], numb_in_prod_block[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_block[non_z_ind], numb_in_prod_block[non_z_ind])
            to_dissolve_block = np.concatenate((to_dissolve_block, repeated_coords), axis=1)
            probs_block = np.concatenate((probs_block, repeated_probs))
            randomise = np.random.random_sample(len(to_dissolve_block[0]))
            temp_ind = np.where(randomise < probs_block)[0]
            to_dissolve_block = to_dissolve_block[:, temp_ind]

            probs_no_neigh = self.cur_case.dissolution_probabilities.dissol_prob.values_pp[to_dissol_no_neigh[2]]
            randomise = np.random.random_sample(len(to_dissol_no_neigh[0]))
            temp_ind = np.where(randomise < probs_no_neigh)[0]
            to_dissol_no_neigh = to_dissol_no_neigh[:, temp_ind]

            to_dissolve = np.concatenate((to_dissolve_no_block, to_dissol_no_neigh, to_dissolve_block), axis=1)

            self.coord_buffer.reset_buffer()
            self.to_dissol_pn_buffer.reset_buffer()

            if len(to_dissolve[0]) > 0:
                just_decrease_counts(self.primary_product.c3d, to_dissolve)
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                insert_counts(self.primary_active.c3d, to_dissolve)
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL(self):
        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, self.comb_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.comb_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            flat_arounds = self.utils.calc_sur_ind_decompose_flat_with_zero(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, flat_arounds)
            all_neigh[:, 6] -= 1

            all_neigh_block = np.array([])
            all_neigh_no_block = np.array([])
            numb_in_prod_block = np.array([], dtype=int)
            numb_in_prod_no_block = np.array([], dtype=int)
            ind_to_raise = np.array([], dtype=int)

            where_not_null = np.unique(np.where(all_neigh[:, :6] > 0)[0])
            to_dissol_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_not_null), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_not_null))

            if self.coord_buffer.last_in_buffer > 0:
                all_neigh = all_neigh[where_not_null]
                numb_in_prod = all_neigh[:, -1]

                all_neigh_bool = np.array(all_neigh[:, :6], dtype=bool)

                arr_len_flat = np.sum(all_neigh_bool, axis=1)

                index_outside = np.where((arr_len_flat < 6))[0]
                self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))

                all_neigh_bool = all_neigh_bool[index_outside]
                arr_len_flat = arr_len_flat[index_outside]
                numb_in_prod = numb_in_prod[index_outside]

                non_flat_arounds = self.utils.calc_sur_ind_decompose_no_flat(self.coord_buffer.get_buffer())
                non_flat_neigh = go_around_bool(self.primary_product.c3d, non_flat_arounds)
                all_neigh_bool = np.concatenate((all_neigh_bool, non_flat_neigh), axis=1)

                ind_where_blocks = aggregate(self.aggregated_ind, all_neigh_bool)

                if len(ind_where_blocks) > 0:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_instead_ind(ind_where_blocks))
                    all_neigh_no_block = np.delete(arr_len_flat, ind_where_blocks)

                    numb_in_prod_no_block = np.delete(numb_in_prod, ind_where_blocks, axis=0)
                    self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(ind_where_blocks))
                    all_neigh_block = arr_len_flat[ind_where_blocks]

                    numb_in_prod_block = numb_in_prod[ind_where_blocks]
                else:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_buffer())
                    all_neigh_no_block = arr_len_flat

                    numb_in_prod_no_block = numb_in_prod

                    self.coord_buffer.reset_buffer()
                    all_neigh_block = np.array([])
                    numb_in_prod_block = np.array([], dtype=int)

            to_dissolve_no_block = self.to_dissol_pn_buffer.get_buffer()
            probs_no_block = self.cur_case.dissolution_probabilities.get_probabilities(all_neigh_no_block,
                                                                                       to_dissolve_no_block[2])

            non_z_ind = np.where(numb_in_prod_no_block != 0)[0]
            repeated_coords = np.repeat(to_dissolve_no_block[:, non_z_ind], numb_in_prod_no_block[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_no_block[non_z_ind], numb_in_prod_no_block[non_z_ind])
            to_dissolve_no_block = np.concatenate((to_dissolve_no_block, repeated_coords), axis=1)
            probs_no_block = np.concatenate((probs_no_block, repeated_probs))
            randomise = np.random.random_sample(len(to_dissolve_no_block[0]))
            temp_ind = np.where(randomise < probs_no_block)[0]
            to_dissolve_no_block = to_dissolve_no_block[:, temp_ind]

            to_dissolve_block = self.coord_buffer.get_buffer()
            probs_block = self.cur_case.dissolution_probabilities.get_probabilities_block(all_neigh_block,
                                                                                          to_dissolve_block[2])
            non_z_ind = np.where(numb_in_prod_block != 0)[0]
            repeated_coords = np.repeat(to_dissolve_block[:, non_z_ind], numb_in_prod_block[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_block[non_z_ind], numb_in_prod_block[non_z_ind])
            to_dissolve_block = np.concatenate((to_dissolve_block, repeated_coords), axis=1)
            probs_block = np.concatenate((probs_block, repeated_probs))
            randomise = np.random.random_sample(len(to_dissolve_block[0]))
            temp_ind = np.where(randomise < probs_block)[0]
            to_dissolve_block = to_dissolve_block[:, temp_ind]

            probs_no_neigh = self.cur_case.dissolution_probabilities.dissol_prob.values_pp[to_dissol_no_neigh[2]]
            randomise = np.random.random_sample(len(to_dissol_no_neigh[0]))
            temp_ind = np.where(randomise < probs_no_neigh)[0]
            to_dissol_no_neigh = to_dissol_no_neigh[:, temp_ind]

            to_dissolve = np.concatenate((to_dissolve_no_block, to_dissol_no_neigh, to_dissolve_block), axis=1)

            self.coord_buffer.reset_buffer()
            self.to_dissol_pn_buffer.reset_buffer()

            if len(to_dissolve[0]) > 0:
                just_decrease_counts(self.primary_product.c3d, to_dissolve)
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                insert_counts(self.primary_active.c3d, to_dissolve)
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def dissolution_zhou_wei_no_bsf(self):
        """
        Implementation of adjusted Zhou and Wei approach. Only side neighbours are checked. No need for block scale
        factor. Works only for any oxidation nuber!
        """
        nz_ind = np.array(np.nonzero(self.cur_case.product.c3d[:, :, self.comb_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.comb_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            all_arounds = self.utils.calc_sur_ind_decompose_flat(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, all_arounds)

            all_neigh_pn = all_neigh[[]]

            # choose all the coordinates which have at least one full side neighbour
            where_full = np.unique(np.where(all_neigh == self.primary_oxid_numb)[0])

            to_dissol_pn_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_full), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_full))

            if self.coord_buffer.last_in_buffer > 0:
                all_neigh = all_neigh[where_full]

                arr_len_flat = np.array([np.sum(item) for item in all_neigh], dtype=np.ubyte)
                index_outside = np.where((arr_len_flat < self.max_inside_neigh_number))[0]

                self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
                all_neigh_pn = arr_len_flat[index_outside]
            else:
                all_neigh_pn = np.array([np.sum(item) for item in all_neigh_pn])

            probs_pn_no_neigh = self.cur_case.dissolution_probabilities.dissol_prob.values_pp[to_dissol_pn_no_neigh[2]]

            to_dissolve_pn = self.to_dissol_pn_buffer.get_buffer()
            probs_pn = self.cur_case.dissolution_probabilities.get_probabilities(all_neigh_pn, to_dissolve_pn[2])

            randomise = np.random.random_sample(len(to_dissol_pn_no_neigh[0]))
            temp_ind = np.where(randomise < probs_pn_no_neigh)[0]
            to_dissol_pn_no_neigh = to_dissol_pn_no_neigh[:, temp_ind]

            randomise = np.random.random_sample(len(to_dissolve_pn[0]))
            temp_ind = np.where(randomise < probs_pn)[0]
            to_dissolve_pn = to_dissolve_pn[:, temp_ind]

            to_dissolve = np.concatenate((to_dissolve_pn, to_dissol_pn_no_neigh), axis=1)

            self.coord_buffer.reset_buffer()
            self.to_dissol_pn_buffer.reset_buffer()

            if len(to_dissolve[0]) > 0:
                counts = self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]]
                self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = 0
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                self.primary_active.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] += counts
                # self.primary_active.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]+1] += counts
                to_dissolve = np.repeat(to_dissolve, counts, axis=1)
                # to_dissolve[2, :] -= 1
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def dissolution_zhou_wei_no_bsf_also_partial_neigh_aip(self):
        """
        Implementation of adjusted Zhou and Wei approach. Only side neighbours are checked. No need for block scale
        factor. Works for oxidation nuber > 1!
        aip: Adjusted Inside Product!
        Im Gegensatz zu dissolution_zhou_wei_no_bsf werden auch die parziellen Nachbarn (weniger als oxidation numb inside)
        berücksichtigt!
        Resolution inside a product: probability for each partial product adjusted according to a number of neighbours
        """
        nz_ind = np.array(np.nonzero(self.cur_case.product.c3d[:, :, self.comb_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.comb_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            all_arounds = self.utils.calc_sur_ind_decompose_flat_with_zero(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, all_arounds)
            all_neigh[:, 6] -= 1

            all_neigh_pn = np.array([])
            numb_in_prod = np.array([], dtype=int)

            where_not_null = np.unique(np.where(all_neigh > 0)[0])
            to_dissol_pn_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_not_null), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_not_null))

            if self.coord_buffer.last_in_buffer > 0:
                all_neigh = all_neigh[where_not_null]
                numb_in_prod = all_neigh[:, 6]

                arr_len_flat = np.sum(all_neigh[:, :6], axis=1)
                index_outside = np.where((arr_len_flat < self.max_inside_neigh_number))[0]

                self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
                all_neigh_pn = arr_len_flat[index_outside]
                numb_in_prod = np.array(numb_in_prod[index_outside], dtype=int)

            to_dissolve_pn = self.to_dissol_pn_buffer.get_buffer()
            probs_pn = self.cur_case.dissolution_probabilities.get_probabilities(all_neigh_pn, to_dissolve_pn[2])

            non_z_ind = np.where(numb_in_prod != 0)[0]
            repeated_coords = np.repeat(to_dissolve_pn[:, non_z_ind], numb_in_prod[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_pn[non_z_ind], numb_in_prod[non_z_ind])

            to_dissolve_pn = np.concatenate((to_dissolve_pn, repeated_coords), axis=1)
            probs_pn = np.concatenate((probs_pn, repeated_probs))

            randomise = np.random.random_sample(len(to_dissolve_pn[0]))
            temp_ind = np.where(randomise < probs_pn)[0]
            to_dissolve_pn = to_dissolve_pn[:, temp_ind]

            probs_pn_no_neigh = self.cur_case.dissolution_probabilities.dissol_prob.values_pp[to_dissol_pn_no_neigh[2]]
            randomise = np.random.random_sample(len(to_dissol_pn_no_neigh[0]))
            temp_ind = np.where(randomise < probs_pn_no_neigh)[0]
            to_dissol_pn_no_neigh = to_dissol_pn_no_neigh[:, temp_ind]

            to_dissolve = np.concatenate((to_dissolve_pn, to_dissol_pn_no_neigh), axis=1)

            self.coord_buffer.reset_buffer()
            self.to_dissol_pn_buffer.reset_buffer()

            if len(to_dissolve[0]) > 0:
                just_decrease_counts(self.primary_product.c3d, to_dissolve)
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                insert_counts(self.primary_active.c3d, to_dissolve)
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def dissolution_simple_with_pd(self):
        """
        Implementation of a simple dissolution approach with single pd for dissolution. No side neighbours are checked,
        no block scale factor, no p_block.
        Works only for any oxidation nuber!
        """
        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, self.product_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.product_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            to_dissolve = self.coord_buffer.get_buffer()
            probs = np.full(len(to_dissolve[0]), self.disol_p)

            randomise = np.random.random_sample(len(to_dissolve[0]))
            temp_ind = np.where(randomise < probs)[0]
            to_dissolve = to_dissolve[:, temp_ind]

            self.coord_buffer.reset_buffer()

            if len(to_dissolve[0]) > 0:
                counts = self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]]
                self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = 0
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                self.primary_active.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] += counts
                to_dissolve = np.repeat(to_dissolve, counts, axis=1)
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def get_combi_ind_standard(self):
        self.ioz_bound = self.get_cur_ioz_bound()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.ioz_bound + 1)], dtype=np.uint32)

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.ioz_bound + 1)], dtype=np.uint32)

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]

        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            self.comb_indexes = oxidant_indexes[indexs]
        else:
            self.comb_indexes = [self.ioz_bound]

    def get_combi_ind_cells_around_product(self):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)

        # self.product_indexes = np.where((product_c < self.param["phase_fraction_lim"]) & (product_c > 0))[0]
        self.product_indexes = np.where(product > 0)[0]
        prod_left_shift = self.product_indexes - 1
        prod_right_shift = self.product_indexes + 1
        self.product_indexes = np.unique(np.concatenate((self.product_indexes, prod_left_shift, prod_right_shift)))
        temp_ind = np.where((self.product_indexes >= 0) & (self.product_indexes <= self.furthest_index))
        self.product_indexes = self.product_indexes[temp_ind]

        # some = np.where((product_c[self.product_indexes] < self.param["phase_fraction_lim"]) & (product_c[self.product_indexes] > 0))[0]
        # some = np.where(product_c[self.product_indexes] < self.param["phase_fraction_lim"])[0]
        # self.product_indexes = self.product_indexes[some]

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]
        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            comb_indexes = oxidant_indexes[indexs]
            self.comb_indexes = np.intersect1d(comb_indexes, self.product_indexes)
        else:
            self.comb_indexes = [self.furthest_index]

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
        # less_than_zero = np.where(matrix_moles < 0)[0]
        # matrix_moles[less_than_zero] = 0
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        if self.iteration % Config.STRIDE == 0:
            self.record_prod_per_layer(self.ioz_bound, product_c, np.zeros(product_c.shape))

        self.product_indexes = np.where(product_c <= Config.PHASE_FRACTION_LIMIT)[0]

        self.comb_indexes = self.get_active_oxidant_mutual_indexes(oxidant, active)
        self.comb_indexes = np.intersect1d(self.comb_indexes, self.product_indexes)

    def get_combi_ind_atomic_no_growth(self):
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
        self.product_indexes = np.where(product_c <= Config.PHASE_FRACTION_LIMIT)[0]

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]
        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            comb_indexes = oxidant_indexes[indexs]
            self.comb_indexes = np.intersect1d(comb_indexes, self.product_indexes)
        else:
            self.comb_indexes = [self.furthest_index]

    def get_combi_ind_two_products(self, current_active):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.ioz_bound + 1)], dtype=np.uint32)
        active = np.array([np.sum(current_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.ioz_bound + 1)], dtype=np.uint32)
        self.comb_indexes = self.get_active_oxidant_mutual_indexes(oxidant, active)

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
        """
        Created only for tests of the solubility product probability function
        """
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

        self.gamma_primes = (((((oxidant_c ** 3) * (active_c ** 2)) / Config.SOL_PROD) - 1) /
                             Config.GENERATED_VALUES.max_gamma_min_one)

        where_solub_prod = np.where(self.gamma_primes > 0)[0]

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]
        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            self.comb_indexes = oxidant_indexes[indexs]
        else:
            self.comb_indexes = [self.furthest_index]

        self.comb_indexes = np.intersect1d(self.comb_indexes, where_solub_prod)

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
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL *\
                               Config.PRODUCTS.PRIMARY.THRESHOLD_OUTWARD

        secondary_product = np.array([np.sum(self.secondary_product.c3d[:, :, plane_ind]) for plane_ind
                                      in range(self.ioz_bound + 1)], dtype=np.uint32)
        secondary_product_moles = secondary_product * Config.PRODUCTS.SECONDARY.MOLES_PER_CELL_TC
        secondary_product_eq_mat_moles = secondary_product * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL *\
                                         Config.PRODUCTS.SECONDARY.THRESHOLD_OUTWARD


        ternary_product = np.array([np.sum(self.ternary_product.c3d[:, :, plane_ind]) for plane_ind
                                      in range(self.ioz_bound + 1)], dtype=np.uint32)
        ternary_product_moles = ternary_product * Config.PRODUCTS.TERNARY.MOLES_PER_CELL_TC
        ternary_product_eq_mat_moles = (ternary_product * ((Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL *
                                        Config.PRODUCTS.TERNARY.THRESHOLD_OUTWARD) + Config.PRODUCTS.TERNARY.MOLES_PER_CELL))


        quaternary_product = np.array([np.sum(self.quaternary_product.c3d[:, :, plane_ind]) for plane_ind
                                    in range(self.ioz_bound + 1)], dtype=np.uint32)
        quaternary_product_moles = quaternary_product * Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL_TC
        quaternary_product_eq_mat_moles = (quaternary_product * ((Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL *
                                           Config.PRODUCTS.QUATERNARY.THRESHOLD_OUTWARD) + Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL))


        quint_product = np.array([np.sum(self.quint_product.c3d[:, :, plane_ind]) for plane_ind
                                       in range(self.ioz_bound + 1)], dtype=np.uint32)
        quint_eq_mat_moles = quint_product * Config.PRODUCTS.QUINT.MOLES_PER_CELL
        quint_product_moles = quint_product * Config.PRODUCTS.QUINT.MOLES_PER_CELL_TC


        matrix_moles = (self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles -
                        secondary_outward_eq_mat_moles - secondary_product_eq_mat_moles - ternary_product_eq_mat_moles -
                        quaternary_product_eq_mat_moles - quint_eq_mat_moles)

        # neg_ind = np.where(matrix_moles < 0)[0]
        # matrix_moles[neg_ind] = 0

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
        # neg_ind = np.where(matrix_moles_pure < 0)[0]
        # matrix_moles_pure[neg_ind] = 0
        whole_moles_pure = matrix_moles_pure + oxidant_pure_moles + active_pure_moles + secondary_active_pure_moles

        oxidant_pure_c = oxidant_pure_moles * 100 / whole_moles_pure
        active_pure_c = active_pure_moles * 100 / whole_moles_pure
        secondary_active_pure_c = secondary_active_pure_moles * 100 / whole_moles_pure

        curr_look_up = self.TdDATA.get_look_up_data(active_pure_c, secondary_active_pure_c, oxidant_pure_c)

        primary_diff = curr_look_up[0] - product_c
        primary_pos_ind = np.where(primary_diff > 0)[0]
        primary_neg_ind = np.where(primary_diff < 0)[0]

        secondary_diff = curr_look_up[1] - secondary_product_c
        secondary_pos_ind = np.where(secondary_diff > 0)[0]
        secondary_neg_ind = np.where(secondary_diff < 0)[0]

        ternary_diff = curr_look_up[2] - ternary_product_c
        ternary_pos_ind = np.where(ternary_diff > 0)[0]
        ternary_neg_ind = np.where(ternary_diff < 0)[0]

        quaternary_diff = curr_look_up[3] - quaternary_product_c
        quaternary_pos_ind = np.where(quaternary_diff > 0)[0]
        quaternary_neg_ind = np.where(quaternary_diff < 0)[0]

        quint_diff = curr_look_up[4] - quint_product_c
        quint_pos_ind = np.where(quint_diff > 0)[0]
        quint_neg_ind = np.where(quint_diff < 0)[0]

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
                # self.decomposition_intrinsic()

        if len(primary_neg_ind) > 0:
            self.comb_indexes = primary_neg_ind
            adj_coeff = (primary_diff[primary_neg_ind] / product_c[primary_neg_ind]) * -1
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes, adj_coeff)
            self.decomposition_intrinsic()
            # self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes,
            #                                                             np.zeros(len(self.comb_indexes)))

        self.cur_case = self.cases.second
        self.cur_case_mp = self.cases.second_mp
        if len(secondary_pos_ind) > 0:
            self.get_combi_ind_two_products(self.secondary_active)
            self.comb_indexes = np.intersect1d(secondary_pos_ind, self.comb_indexes)

            if len(self.comb_indexes) > 0:
                self.cases.reaccumulate_products(self.cur_case)
                self.precip_mp()

        self.comb_indexes = np.arange(self.ioz_bound + 1)
        self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes, np.zeros(len(self.comb_indexes)))
        self.decomposition_intrinsic()

        if len(secondary_neg_ind) > 0:
            self.comb_indexes = secondary_neg_ind
            adj_coeff = (secondary_diff[secondary_neg_ind] / secondary_product_c[secondary_neg_ind]) * -1
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes, adj_coeff)
            self.decomposition_intrinsic()
            # self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes,
            #                                                             np.zeros(len(self.comb_indexes)))

        self.cur_case = self.cases.third
        self.cur_case_mp = self.cases.third_mp
        if len(ternary_pos_ind) > 0:
            self.get_combi_ind_two_products(self.primary_active)
            self.comb_indexes = np.intersect1d(ternary_pos_ind, self.comb_indexes)

            if len(self.comb_indexes) > 0:
                self.cases.reaccumulate_products(self.cur_case)
                self.precip_mp()
                # self.decomposition_intrinsic()

        if len(ternary_neg_ind) > 0:
            self.comb_indexes = ternary_neg_ind
            adj_coeff = (ternary_diff[ternary_neg_ind] / ternary_product_c[ternary_neg_ind]) * -1
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes, adj_coeff)
            self.decomposition_intrinsic()
            # self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes,
            #                                                                np.zeros(len(self.comb_indexes)))

        self.cur_case = self.cases.fourth
        self.cur_case_mp = self.cases.fourth_mp
        if len(quaternary_pos_ind) > 0:
            self.get_combi_ind_two_products(self.secondary_active)
            self.comb_indexes = np.intersect1d(quaternary_pos_ind, self.comb_indexes)

            if len(self.comb_indexes) > 0:
                self.cases.reaccumulate_products(self.cur_case)
                self.precip_mp()
                # self.decomposition_intrinsic()

        if len(quaternary_neg_ind) > 0:
            self.comb_indexes = quaternary_neg_ind
            adj_coeff = (quaternary_diff[quaternary_neg_ind] / quaternary_product_c[quaternary_neg_ind]) * -1
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes, adj_coeff)
            self.decomposition_intrinsic()
            # self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes,
            #                                                                np.zeros(len(self.comb_indexes)))

        self.cur_case = self.cases.fifth
        self.cur_case_mp = self.cases.fifth_mp
        if len(quint_pos_ind) > 0:
            self.comb_indexes = quint_pos_ind

            if len(self.comb_indexes) > 0:
                self.cases.reaccumulate_products(self.cur_case)
                self.precip_mp()
                # self.decomposition_intrinsic()

        if len(quint_neg_ind) > 0:
            self.comb_indexes = quint_neg_ind
            adj_coeff = (quint_diff[quint_neg_ind] / quint_product_c[quint_neg_ind]) * -1
            self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes, adj_coeff)
            self.decomposition_intrinsic()
            # self.cur_case_mp.dissolution_probabilities.adapt_probabilities(self.comb_indexes,
            #                                                                np.zeros(len(self.comb_indexes)))

    def precipitation_with_td(self):
        self.furthest_index = self.primary_oxidant.calc_furthest_index()

        if self.furthest_index >= self.curr_max_furthest:
            self.curr_max_furthest = self.furthest_index

        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            self.primary_active.transform_to_3d(self.curr_max_furthest)
            self.secondary_active.transform_to_3d(self.curr_max_furthest)

        self.calc_stable_products_all()
        self.primary_oxidant.transform_to_descards()


    def precipitation_first_case(self):
        # Only one oxidant and one active elements exist. Only one product can be created
        self.furthest_index = self.primary_oxidant.calc_furthest_index()
        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            if self.furthest_index >= self.curr_max_furthest:
                self.curr_max_furthest = self.furthest_index
            self.primary_active.transform_to_3d(self.curr_max_furthest)

        self.get_combi_ind()

        if len(self.comb_indexes) > 0:
            # if len(self.comb_indexes) > self.prev_len:
            #     print("New depth: ", self.comb_indexes)
            #     self.prev_len = len(self.comb_indexes)

            self.cur_case = self.cases.first
            self.cur_case_mp = self.cases.first_mp

            self.precip_mp()

        self.primary_oxidant.transform_to_descards()

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

            # if len(self.comb_indexes) > self.prev_len:
            #     print("New depth: ", self.comb_indexes)
            #     self.prev_len = len(self.comb_indexes)

            self.cur_case = self.cases.first
            self.cur_case_mp = self.cases.first_mp

            self.precip_mp()

        if len(saved_ind) > 0:
            self.comb_indexes = saved_ind
            # if len(self.comb_indexes) > self.prev_len:
            #     print("New depth: ", self.comb_indexes)
            #     self.prev_len = len(self.comb_indexes)

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

    def dissolution_atomic_stop_if_no_active(self):
        self.ioz_bound = self.ioz_depth_from_kinetics()
        self.comb_indexes = np.where(self.cur_case.prod_indexes)[0]

        temp = np.where(self.comb_indexes <= self.ioz_bound)[0]
        self.comb_indexes = self.comb_indexes[temp]

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind in self.comb_indexes], dtype=np.uint32)

        # product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
        #                     in self.product_indexes], dtype=np.uint32)
        # temp_ind = np.where(product == 0)[0]

        # self.product_x_nzs[self.product_indexes[temp_ind]] = False
        # active = np.delete(active, temp_ind)
        # self.comb_indexes = np.delete(self.product_indexes, temp_ind)

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

            results = self.pool.map(worker, tasks)

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

    def dissolution_test(self):
        self.comb_indexes = self.get_cur_dissol_ioz_bound()

        self.comb_indexes = np.where(self.cur_case.prod_indexes)[0]
        temp = np.where(self.comb_indexes <= self.ioz_bound)[0]
        self.comb_indexes = self.comb_indexes[temp]

        # not_stable_ind = np.where(self.product_x_not_stab)[0]
        # nz_ind = np.where(self.product_x_nzs)[0]

        # self.product_indexes = np.intersect1d(not_stable_ind, nz_ind)

        # product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
        #                     in self.product_indexes], dtype=np.uint32)

        # where_no_prod = np.where(product == 0)[0]
        # self.product_x_nzs[self.product_indexes[where_no_prod]] = False

        # self.product_indexes = np.where(self.product_x_nzs)[0]

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()
        # else:
        #     print("PRODUCT FULLY DISSOLVED AFTER ", self.iteration, " ITERATIONS")
        #     sys.exit()

    def precip_step_standard(self):
        for plane_index in reversed(self.comb_indexes):
            for fetch_ind in self.fetch_ind:
                oxidant_cells = self.cur_case.oxidant.c3d[fetch_ind[0], fetch_ind[1], plane_index]
                oxidant_cells = fetch_ind[:, np.nonzero(oxidant_cells)[0]]

                if len(oxidant_cells[0]) != 0:
                    oxidant_cells = np.vstack((oxidant_cells, np.full(len(oxidant_cells[0]), plane_index,
                                                                      dtype=np.short)))
                    oxidant_cells = oxidant_cells.transpose()

                    exists = check_at_coord(self.cur_case.product.full_c3d, oxidant_cells)  # precip on place of oxidant!
                    temp_ind = np.where(exists)[0]
                    oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

                    if len(oxidant_cells) > 0:
                        # activate if microstructure ___________________________________________________________
                        # in_gb = [self.microstructure[point[0], point[1], point[2]] for point in oxidant_cells]
                        # temp_ind = np.where(in_gb)[0]
                        # oxidant_cells = oxidant_cells[temp_ind]
                        # ______________________________________________________________________________________
                        self.check_intersection(oxidant_cells)

    def precip_step_no_growth(self):
        for plane_index in reversed(self.comb_indexes):
            for fetch_ind in self.fetch_ind:
                oxidant_cells = self.cur_case.oxidant.c3d[fetch_ind[0], fetch_ind[1], plane_index]
                oxidant_cells = fetch_ind[:, np.nonzero(oxidant_cells)[0]]

                if len(oxidant_cells[0]) != 0:
                    oxidant_cells = np.vstack((oxidant_cells, np.full(len(oxidant_cells[0]), plane_index, dtype=np.short)))
                    oxidant_cells = oxidant_cells.transpose()

                    exists = check_at_coord(self.cur_case.product.full_c3d, oxidant_cells)  # precip on place of oxidant!
                    temp_ind = np.where(exists)[0]
                    oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

                    if len(oxidant_cells) > 0:
                        self.check_intersection(oxidant_cells)

    def precip_step_no_growth_solub_prod_test(self):
        """
        Created only for tests of the solubility product probability function
        """
        for plane_index in reversed(self.comb_indexes):
            for fetch_ind in self.fetch_ind:
                oxidant_cells = self.cur_case.oxidant.c3d[fetch_ind[0], fetch_ind[1], plane_index]
                oxidant_cells = fetch_ind[:, np.nonzero(oxidant_cells)[0]]

                if len(oxidant_cells[0]) != 0:
                    oxidant_cells = np.vstack((oxidant_cells, np.full(len(oxidant_cells[0]), plane_index, dtype=np.short)))
                    oxidant_cells = oxidant_cells.transpose()

                    exists = check_at_coord(self.cur_case.product.full_c3d, oxidant_cells)  # precip on place of oxidant!
                    temp_ind = np.where(exists)[0]
                    oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

                    if len(oxidant_cells) > 0:
                        self.ci_single_no_growth_solub_prod_test(oxidant_cells)

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

    def diffusion_inward(self):
        # self.cases.reaccumulate_products_no_exclusion()
        self.primary_oxidant.diffuse()
        if Config.OXIDANTS.SECONDARY_EXISTENCE:
            self.secondary_oxidant.diffuse()

    def diffusion_outward(self):
        if (self.iteration + 1) % Config.STRIDE == 0:
            self.cur_case = self.cases.first
            self.cur_case_mp = self.cases.first_mp
            self.diffusion_outward_mp()
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.cur_case = self.cases.second
                self.cur_case_mp = self.cases.second_mp
                self.diffusion_outward_mp()

    def diffusion_outward_mp(self):
        if (self.iteration + 1) % Config.STRIDE == 0:

            self.cur_case.active.transform_to_descards()  # UNCOMMENT!!!!!!!!!!

            chunk_size = self.cur_case.active.last_in_diff_arr // self.numb_of_proc
            remainder = self.cur_case.active.last_in_diff_arr % self.numb_of_proc
            indices = []
            start = 0
            for i in range(self.numb_of_proc):
                end = start + chunk_size + (1 if i < remainder else 0)
                indices.append([start, end])
                start = end

            tasks = [(wr, self.cur_case_mp, self.cur_case.active.p_ranges, self.cur_case.active.diffuse) for wr in indices]
            results = self.pool.map(worker, tasks)
            to_del = np.array(np.concatenate(results))
            self.cur_case.active.dell_cells_from_diff_arrays(to_del)
            self.cur_case.active.fill_first_page()

    def diffusion_outward_with_mult_srtide(self):
        if self.iteration % Config.STRIDE == 0:
            if self.iteration % self.precipitation_stride == 0 or self.iteration == 0:
                self.primary_active.transform_to_descards()
            self.primary_active.diffuse()
            # if Config.ACTIVES.SECONDARY_EXISTENCE:
            #     self.secondary_active.transform_to_descards()
            #     self.secondary_active.diffuse()

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
        self.cur_case.precip_3d_init[:, :, 0:u_bound + 2] = 0
        self.cur_case.precip_3d_init[:, :, 0:u_bound + 2] = self.cur_case.product.c3d[:, :, 0:u_bound + 2]

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

    def generate_fetch_ind(self):
        size = 3 + (Config.NEIGH_RANGE - 1) * 2
        if self.cells_per_axis % size == 0:
            length = int((self.cells_per_axis / size) ** 2)
            self.fetch_ind = np.zeros((size**2, 2, length), dtype=np.short)
            iter_shifts = np.array(np.where(np.ones((size, size)) == 1)).transpose()
            dummy_grid = np.full((self.cells_per_axis, self.cells_per_axis), True)
            all_coord = np.array(np.nonzero(dummy_grid), dtype=np.short)
            for step, t in enumerate(iter_shifts):
                t_ind = np.where(((all_coord[0] - t[1]) % size == 0) & ((all_coord[1] - t[0]) % size == 0))[0]
                self.fetch_ind[step] = all_coord[:, t_ind]
            self.fetch_ind = np.array(self.fetch_ind, dtype=np.ushort)
        else:
            print()
            print("______________________________________________________________")
            print("Number of Cells per Axis must be divisible by ", size, "!!!")
            print("______________________________________________________________")
            sys.exit()

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

    def generate_fetch_ind_mp(self):
        size = 3 + (Config.NEIGH_RANGE - 1) * 2
        if Config.N_CELLS_PER_AXIS % size == 0:
            numb_of_div_per_page = Config.NUMBER_OF_DIVS_PER_PAGE

            if numb_of_div_per_page > 1:
                p_chunk_size = int((Config.N_CELLS_PER_AXIS / numb_of_div_per_page) - Config.NEIGH_RANGE * 2)
                s_chunk_size = Config.NEIGH_RANGE * 2

                p_chunk_ranges = np.zeros((numb_of_div_per_page, 2), dtype=int)
                p_chunk_ranges[0] = [Config.NEIGH_RANGE, Config.NEIGH_RANGE + p_chunk_size]

                for pos in range(1, numb_of_div_per_page):
                    p_chunk_ranges[pos, 0] = p_chunk_ranges[pos - 1, 1] + s_chunk_size
                    p_chunk_ranges[pos, 1] = p_chunk_ranges[pos, 0] + p_chunk_size
                p_chunk_ranges[-1, 1] = Config.N_CELLS_PER_AXIS - Config.NEIGH_RANGE

                s_chunk_ranges = np.zeros((numb_of_div_per_page + 1, 2), dtype=int)
                s_chunk_ranges[0] = [0, Config.NEIGH_RANGE]

                for pos in range(1, numb_of_div_per_page + 1):
                    s_chunk_ranges[pos, 0] = s_chunk_ranges[pos - 1, 1] + p_chunk_size
                    s_chunk_ranges[pos, 1] = s_chunk_ranges[pos, 0] + s_chunk_size

                s_chunk_ranges[-1, 1] = Config.N_CELLS_PER_AXIS
                s_chunk_ranges[-1, 0] = p_chunk_ranges[-1, 1]

                for item in p_chunk_ranges:
                    new_batch = self.generate_batch_fetch_ind_mp(item, size)
                    self.primary_fetch_ind.append(new_batch)

                f_and_l = self.generate_batch_fetch_ind_mp([s_chunk_ranges[0], s_chunk_ranges[-1]], size, switch=True)
                self.secondary_fetch_ind.append(f_and_l)
                for index, item in enumerate(s_chunk_ranges):
                    if index == 0 or index == len(s_chunk_ranges) - 1:
                        continue
                    new_batch = self.generate_batch_fetch_ind_mp(item, size)
                    self.secondary_fetch_ind.append(new_batch)
            else:
                p_chunk_ranges = np.array([[0, Config.N_CELLS_PER_AXIS]], dtype=int)

                for item in p_chunk_ranges:
                    new_batch = self.generate_batch_fetch_ind_mp(item, size)
                    self.primary_fetch_ind.append(new_batch)

            # dummy_grid1 = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS), False, dtype=bool)
            #
            # for item in self.primary_fetch_ind:
            #     for coord_set in item:
            #         for ind in range(len(coord_set[0])):
            #             z_coord = coord_set[0, ind]
            #             y_coord = coord_set[1, ind]
            #             if dummy_grid1[z_coord, y_coord]:
            #                 print("ALLREADY TRUE AT: ", z_coord, " ", y_coord)
            #             else:
            #                 dummy_grid1[z_coord, y_coord] = True
            # print()
            #
            # dummy_grid2 = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS), False, dtype=bool)
            #
            # for item in self.secondary_fetch_ind:
            #     for coord_set in item:
            #         for ind in range(len(coord_set[0])):
            #             z_coord = coord_set[0, ind]
            #             y_coord = coord_set[1, ind]
            #             if dummy_grid2[z_coord, y_coord]:
            #                 print("ALLREADY TRUE AT: ", z_coord, " ", y_coord)
            #             else:
            #                 dummy_grid2[z_coord, y_coord] = True
            # print()

        else:
            print()
            print("______________________________________________________________")
            print("Number of Cells per Axis must be divisible by ", size, "!!!")
            print("______________________________________________________________")
            sys.exit()

    def record_prod_per_layer(self, ioz_bound, product_is, product_goal):
        itera = np.full(ioz_bound + 1, self.iteration) // Config.STRIDE
        indexes = np.stack((range(ioz_bound + 1), itera))
        self.cumul_prod.set_at_ind(indexes, product_is)
        self.growth_rate.set_at_ind(indexes, product_goal)

    def simple_decompose_mp(self):
        tasks = [(self.cur_case_mp.product_c3d_shm_mdata, chunk_range, self.comb_indexes, self.aggregated_ind,
                  self.cur_case_mp.dissolution_probabilities, self.cur_case_mp.decomposition) for chunk_range in
                 self.chunk_ranges]

        results = self.pool.map(worker, tasks)

        to_dissolve = np.array(np.concatenate(results, axis=1), dtype=np.ushort)
        if len(to_dissolve[0]) > 0:
            just_decrease_counts(self.cur_case.product.c3d, to_dissolve)
            self.cur_case.product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
            if self.cur_case_mp.threshold_outward > 0:
                insert_counts(self.cur_case.active.c3d, to_dissolve, self.cur_case_mp.threshold_outward)
            repeated_coords = np.repeat(to_dissolve, self.cur_case_mp.threshold_inward, axis=1)
            self.cur_case.oxidant.cells = np.concatenate((self.cur_case.oxidant.cells, repeated_coords), axis=1)
            new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(repeated_coords[0]))
            new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
            new_dirs -= 1
            self.cur_case.oxidant.dirs = np.concatenate((self.cur_case.oxidant.dirs, new_dirs), axis=1)

    def precip_mp(self):
        self.cur_case.fix_init_precip_func_ref(self.ioz_bound)
        if len(self.comb_indexes) <= Config.DEPTH_PER_DIV:
            p_tasks = [(self.cur_case_mp, self.comb_indexes, fetch_batch, self.cur_case_mp.check_intersection,
                        self.cur_case_mp.precip_step) for fetch_batch in self.primary_fetch_ind]
            s_tasks = [(self.cur_case_mp, self.comb_indexes, fetch_batch, self.cur_case_mp.check_intersection,
                        self.cur_case_mp.precip_step) for fetch_batch in self.secondary_fetch_ind]
        else:
            ind_chunks = [self.comb_indexes[i:i + Config.DEPTH_PER_DIV]
                          for i in range(0, len(self.comb_indexes), Config.DEPTH_PER_DIV)]
            p_tasks = [(self.cur_case_mp, ind, fetch_batch, self.cur_case_mp.check_intersection,
                        self.cur_case_mp.precip_step) for ind in ind_chunks for fetch_batch in self.primary_fetch_ind]
            s_tasks = [(self.cur_case_mp, ind, fetch_batch, self.cur_case_mp.check_intersection,
                        self.cur_case_mp.precip_step) for ind in ind_chunks for fetch_batch in self.secondary_fetch_ind]
        self.pool.map(worker, p_tasks)
        self.pool.map(worker, s_tasks)

    def ioz_depth_from_kinetics(self):
        self.curr_time = Config.GENERATED_VALUES.TAU * (self.iteration + 1)
        active_ind = np.where(self.active_times <= self.curr_time)[0]
        return min(np.amax(active_ind), self.furthest_index)

    def ioz_depth_furthest_inward(self):
        return self.furthest_index

    def ioz_dissolution_where_prod(self):
        return np.where(self.cur_case.prod_indexes)[0]
