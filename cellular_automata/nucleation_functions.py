from utils.numba_functions import *
from multiprocessing import shared_memory
from .neigh_indexes import *


def precip_step_standard(cur_case, plane_indexes, fetch_indexes, callback):
    shm_o = shared_memory.SharedMemory(name=cur_case.oxidant_c3d_shm_mdata.name)
    oxidant = np.ndarray(cur_case.oxidant_c3d_shm_mdata.shape, dtype=cur_case.oxidant_c3d_shm_mdata.dtype, buffer=shm_o.buf)
    shm_p_FULL = shared_memory.SharedMemory(name=cur_case.full_shm_mdata.name)
    full_3d = np.ndarray(cur_case.full_shm_mdata.shape, dtype=cur_case.full_shm_mdata.dtype, buffer=shm_p_FULL.buf)

    for fetch_ind in fetch_indexes:
        plane_indexes = np.array(plane_indexes)
        nonzero_indices = np.nonzero(oxidant[fetch_ind[0][:, np.newaxis], fetch_ind[1][:, np.newaxis], plane_indexes])
        oxidant_cells = fetch_ind[:, nonzero_indices[0]]

        if len(oxidant_cells[0]) != 0:
            oxidant_cells = np.vstack((oxidant_cells, plane_indexes[np.array(nonzero_indices[1])]))
            oxidant_cells = np.array(oxidant_cells, dtype=np.short).transpose()
            exists = check_at_coord(full_3d, oxidant_cells)  # precip on place of oxidant!
            temp_ind = np.where(exists)[0]
            oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

            if len(oxidant_cells) > 0:
                # activate if microstructure ___________________________________________________________
                # in_gb = [self.microstructure[point[0], point[1], point[2]] for point in oxidant_cells]
                # temp_ind = np.where(in_gb)[0]
                # oxidant_cells = oxidant_cells[temp_ind]
                # ______________________________________________________________________________________
                callback(cur_case, oxidant_cells, oxidant, full_3d)
    shm_o.close()
    shm_p_FULL.close()


def precip_step_multi_products(cur_case, plane_indexes, fetch_indexes, callback):

    shm_o = shared_memory.SharedMemory(name=cur_case.oxidant_c3d_shm_mdata.name)
    oxidant = np.ndarray(cur_case.oxidant_c3d_shm_mdata.shape, dtype=cur_case.oxidant_c3d_shm_mdata.dtype,
                         buffer=shm_o.buf)

    shm_p_FULL = shared_memory.SharedMemory(name=cur_case.full_shm_mdata.name)
    full_3d = np.ndarray(cur_case.full_shm_mdata.shape, dtype=cur_case.full_shm_mdata.dtype, buffer=shm_p_FULL.buf)

    shm_to_check_with = shared_memory.SharedMemory(name=cur_case.to_check_with_shm_mdata.name)
    to_check_with = np.ndarray(cur_case.to_check_with_shm_mdata.shape, dtype=cur_case.to_check_with_shm_mdata.dtype,
                               buffer=shm_to_check_with.buf)

    for fetch_ind in fetch_indexes:
        plane_indexes = np.array(plane_indexes)
        nonzero_indices = np.nonzero(oxidant[fetch_ind[0][:, np.newaxis], fetch_ind[1][:, np.newaxis], plane_indexes])
        oxidant_cells = fetch_ind[:, nonzero_indices[0]]

        if len(oxidant_cells[0]) != 0:
            oxidant_cells = np.vstack((oxidant_cells, plane_indexes[np.array(nonzero_indices[1])]))
            oxidant_cells = np.array(oxidant_cells, dtype=np.short).transpose()

            exists = check_at_coord(full_3d, oxidant_cells)  # precip on place of oxidant!
            temp_ind = np.where(exists)[0]
            oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

            exists = check_at_coord(to_check_with, oxidant_cells)  # precip on place of oxidant!
            temp_ind = np.where(exists)[0]
            oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

            if len(oxidant_cells) > 0:
                # activate if microstructure ___________________________________________________________
                # in_gb = [self.microstructure[point[0], point[1], point[2]] for point in oxidant_cells]
                # temp_ind = np.where(in_gb)[0]
                # oxidant_cells = oxidant_cells[temp_ind]
                # ______________________________________________________________________________________
                callback(cur_case, oxidant_cells, oxidant, full_3d)

    shm_o.close()
    shm_p_FULL.close()
    shm_to_check_with.close()


def ci_single(cur_case, seeds, oxidant, full_3d):
    shm_p = shared_memory.SharedMemory(name=cur_case.product_c3d_shm_mdata.name)
    product = np.ndarray(cur_case.product_c3d_shm_mdata.shape, dtype=cur_case.product_c3d_shm_mdata.dtype, buffer=shm_p.buf)
    shm_a = shared_memory.SharedMemory(name=cur_case.active_c3d_shm_mdata.name)
    active = np.ndarray(cur_case.active_c3d_shm_mdata.shape, dtype=cur_case.active_c3d_shm_mdata.dtype, buffer=shm_a.buf)
    shm_product_init = shared_memory.SharedMemory(name=cur_case.precip_3d_init_shm_mdata.name)
    product_init = np.ndarray(cur_case.precip_3d_init_shm_mdata.shape, dtype=cur_case.precip_3d_init_shm_mdata.dtype,
                              buffer=shm_product_init.buf)
    shm_product_x_nzs = shared_memory.SharedMemory(name=cur_case.prod_indexes_shm_mdata.name)
    product_x_nzs = np.ndarray(cur_case.prod_indexes_shm_mdata.shape, dtype=cur_case.prod_indexes_shm_mdata.dtype,
                               buffer=shm_product_x_nzs.buf)

    all_arounds = calc_sur_ind_formation(seeds, active.shape[2] - 1)
    neighbours = go_around_bool(active, all_arounds[:, :-2])
    arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.short)
    temp_ind = np.where(arr_len_out > 0)[0]

    if len(temp_ind) > 0:
        seeds = seeds[temp_ind]
        neighbours = neighbours[temp_ind]
        all_arounds = all_arounds[temp_ind]

        flat_arounds = np.concatenate((all_arounds[:, 0:5], all_arounds[:, -2:]), axis=1)
        arr_len_in_flat = cur_case.go_around_func_ref(product_init, flat_arounds)

        homogeneous_ind = np.where(arr_len_in_flat == 0)[0]
        needed_prob = cur_case.nucleation_probabilities.get_probabilities(arr_len_in_flat, seeds[:, 2])
        needed_prob[homogeneous_ind] = cur_case.nucleation_probabilities.nucl_prob.values_pp[seeds[homogeneous_ind, 2]]
        randomise = np.array(np.random.random_sample(arr_len_in_flat.size), dtype=np.float64)
        temp_ind = np.where(randomise < needed_prob)[0]

        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]

            out_to_del = [np.array(np.nonzero(item)[0]) for item in neighbours]
            to_del = [np.random.choice(item, 1, replace=False) for item in out_to_del]
            coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                             dtype=np.short)

            coord = np.reshape(coord, (len(coord) * 1, 3))
            coord = coord.transpose()
            seeds = seeds.transpose()

            active[coord[0], coord[1], coord[2]] -= 1
            oxidant[seeds[0], seeds[1], seeds[2]] -= 1

            # self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
            product[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

            # self.objs[self.case]["product"].fix_full_cells(coord)  # precip on place of active!
            # self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!

            cur_case.fix_full_cells(product, full_3d, seeds, cur_case.oxidation_number)

            # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
            # dissolution function
            product_x_nzs[seeds[2][0]] = True
    shm_p.close()
    shm_a.close()
    shm_product_init.close()
    shm_product_x_nzs.close()


def ci_multi(cur_case, seeds, oxidant, full_3d):
    """
    Check intersections between the seeds neighbourhood and the coordinates of inward particles only.
    Compute which seed will become a precipitation and which inward particles should be deleted
    according to threshold_inward conditions. This is a simplified version of the check_intersection() function
    where threshold_outward is equal to 1, so there is no need to check intersection with OUT arrays!

    :param seeds: array of seeds coordinates
    """

    shm_p = shared_memory.SharedMemory(name=cur_case.product_c3d_shm_mdata.name)
    product = np.ndarray(cur_case.product_c3d_shm_mdata.shape, dtype=cur_case.product_c3d_shm_mdata.dtype,
                         buffer=shm_p.buf)
    shm_a = shared_memory.SharedMemory(name=cur_case.active_c3d_shm_mdata.name)
    active = np.ndarray(cur_case.active_c3d_shm_mdata.shape, dtype=cur_case.active_c3d_shm_mdata.dtype,
                        buffer=shm_a.buf)
    shm_product_init = shared_memory.SharedMemory(name=cur_case.precip_3d_init_shm_mdata.name)
    product_init = np.ndarray(cur_case.precip_3d_init_shm_mdata.shape, dtype=cur_case.precip_3d_init_shm_mdata.dtype,
                              buffer=shm_product_init.buf)
    shm_product_x_nzs = shared_memory.SharedMemory(name=cur_case.prod_indexes_shm_mdata.name)
    product_x_nzs = np.ndarray(cur_case.prod_indexes_shm_mdata.shape, dtype=cur_case.prod_indexes_shm_mdata.dtype,
                               buffer=shm_product_x_nzs.buf)

    all_arounds = calc_sur_ind_formation(seeds, active.shape[2] - 1)
    self_neighbours = go_around_int(oxidant, all_arounds[:, :-2])
    self_neighbours[:, 4] -= 1
    self_neighbours = np.array(self_neighbours, dtype=bool)
    arr_len_self = np.array([np.sum(item) for item in self_neighbours], dtype=np.short)
    temp_ind = np.where(arr_len_self >= cur_case.threshold_inward - 1)[0]

    if len(temp_ind) > 0:
        seeds = seeds[temp_ind]
        self_neighbours = self_neighbours[temp_ind]
        all_arounds = all_arounds[temp_ind]

        neighbours = go_around_bool(active, all_arounds[:, :-2])
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.short)
        temp_ind = np.where(arr_len_out >= cur_case.threshold_outward)[0]

        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            self_neighbours = self_neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]

            flat_arounds = np.concatenate((all_arounds[:, 0:5], all_arounds[:, -2:]), axis=1)
            arr_len_in_flat = cur_case.go_around_func_ref(product_init, flat_arounds)

            homogeneous_ind = np.where(arr_len_in_flat == 0)[0]
            needed_prob = cur_case.nucleation_probabilities.get_probabilities(arr_len_in_flat, seeds[:, 2])
            needed_prob[homogeneous_ind] = cur_case.nucleation_probabilities.nucl_prob.values_pp[
                seeds[homogeneous_ind, 2]]
            randomise = np.array(np.random.random_sample(arr_len_in_flat.size), dtype=np.float64)
            temp_ind = np.where(randomise < needed_prob)[0]

            if len(temp_ind) > 0:
                seeds = seeds[temp_ind]
                neighbours = neighbours[temp_ind]
                self_neighbours = self_neighbours[temp_ind]
                all_arounds = all_arounds[temp_ind]

                # out_to_del = np.array(np.nonzero(neighbours))
                # start_seed_index = np.unique(out_to_del[0], return_index=True)[1]
                # to_del = np.array([out_to_del[1, indx:indx + cur_case.threshold_outward] for indx in start_seed_index],
                #                   dtype=int)
                # coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                #                  dtype=np.short)

                out_to_del = [np.array(np.nonzero(item)[0]) for item in neighbours]
                to_del = [np.random.choice(item, cur_case.threshold_outward, replace=False) for item in out_to_del]
                out_coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                                 dtype=np.short)

                in_to_del = [np.array(np.nonzero(item)[0]) for item in self_neighbours]
                to_del = [np.random.choice(item, cur_case.threshold_inward - 1, replace=False) for item in in_to_del]
                in_coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                                 dtype=np.short)

                # coord = np.reshape(coord, (len(coord) * self.threshold_inward, 3)).transpose()
                out_coord = np.reshape(out_coord, (len(out_coord) * cur_case.threshold_outward, 3))
                in_coord = np.reshape(in_coord, (len(in_coord) * cur_case.threshold_inward - 1, 3))

                out_coord = out_coord.transpose()
                in_coord = in_coord.transpose()
                seeds = seeds.transpose()

                active[out_coord[0], out_coord[1], out_coord[2]] -= 1
                oxidant[in_coord[0], in_coord[1], in_coord[2]] -= 1
                oxidant[seeds[0], seeds[1], seeds[2]] -= 1

                # self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
                product[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

                # self.objs[self.case]["product"].fix_full_cells(coord)  # precip on place of active!
                # self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!

                cur_case.fix_full_cells(product, full_3d, seeds, cur_case.oxidation_number)

                # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
                # dissolution function
                product_x_nzs[seeds[2][0]] = True
    shm_p.close()
    shm_a.close()
    shm_product_init.close()
    shm_product_x_nzs.close()

                # exists = [self.cur_case.product.full_c3d[point[0], point[1], point[2]] for point in coord]
                # temp_ind = np.where(exists)[0]
                # coord = np.delete(coord, temp_ind, 0)
                # seeds = np.delete(seeds, temp_ind, 0)
                #
                # if self.cur_case.to_check_with is not None:
                #     exists = [self.cur_case.to_check_with.c3d[point[0], point[1], point[2]] for point in coord]
                #     temp_ind = np.where(exists)[0]
                #     coord = np.delete(coord, temp_ind, 0)
                #     seeds = np.delete(seeds, temp_ind, 0)

                # if len(seeds) > 0:
                #     self_all_arounds = self.utils.calc_sur_ind_formation_noz(seeds, self.cur_case.oxidant.c3d.shape[2] - 1)
                #     self_neighbours = np.array([[self.cur_case.oxidant.c3d[point[0], point[1], point[2]]
                #                                  for point in seed_arrounds]
                #                                 for seed_arrounds in self_all_arounds], dtype=bool)
                #     arr_len_in = np.array([np.sum(item) for item in self_neighbours], dtype=np.ubyte)
                #     temp_ind = np.where(arr_len_in >= self.threshold_inward)[0]
                #     # if len(index_in) > 0:
                #     #     seeds = seeds[index_in]
                #     #     neighbours = neighbours[index_in]
                #     #     all_arounds = all_arounds[index_in]
                #     #     flat_arounds = all_arounds[:, 0:6]
                #     #     flat_neighbours = np.array(
                #     #         [[self.precipitations3d_init[point[0], point[1], point[2]] for point in seed_arrounds]
                #     #          for seed_arrounds in flat_arounds], dtype=bool)
                #     #     arr_len_in_flat = np.array([np.sum(item) for item in flat_neighbours], dtype=int)
                #     #     homogeneous_ind = np.where(arr_len_in_flat == 0)[0]
                #     #     needed_prob = self.const_a * 2.718281828 ** (self.const_b * arr_len_in_flat)
                #     #     needed_prob[homogeneous_ind] = self.scale_probability
                #     #     randomise = np.random.random_sample(len(arr_len_in_flat))
                #     #     temp_ind = np.where(randomise < needed_prob)[0]
                #
                #     if len(temp_ind) > 0:
                #         seeds = seeds[temp_ind]
                #         coord = coord[temp_ind]
                #
                #         # neighbours = neighbours[temp_ind]
                #         # all_arounds = all_arounds[temp_ind]
                #
                #         self_neighbours = self_neighbours[temp_ind]
                #         self_all_arounds = self_all_arounds[temp_ind]
                #
                #         in_to_del = np.array(np.nonzero(self_neighbours))
                #         in_start_seed_index = np.unique(in_to_del[0], return_index=True)[1]
                #         to_del_in = np.array(
                #             [in_to_del[1, indx:indx + self.threshold_inward - 1] for indx in in_start_seed_index],
                #             dtype=int)
                #         coord_in = np.array([self_all_arounds[seed_ind][point_ind] for seed_ind, point_ind in
                #                              enumerate(to_del_in)], dtype=np.short)
                #         coord_in = np.reshape(coord_in, (len(coord_in) * (self.threshold_inward - 1), 3)).transpose()
                #
                #         # out_to_del = np.array(np.nonzero(neighbours))
                #         # start_seed_index = np.unique(out_to_del[0], return_index=True)[1]
                #         # to_del = np.array([out_to_del[1, indx:indx + self.threshold_outward] for indx in start_seed_index],
                #         #                   dtype=int)
                #         # coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                #         #                  dtype=np.short)
                #         # # coord = np.reshape(coord, (len(coord) * self.threshold_inward, 3)).transpose()
                #         # coord = np.reshape(coord, (len(coord) * self.threshold_outward, 3))
                #         # exists = [product.full_c3d[point[0], point[1], point[2]] for point in coord]
                #         # temp_ind = np.where(exists)[0]
                #         # coord = np.delete(coord, temp_ind, 0)
                #         # seeds = np.delete(seeds, temp_ind, 0)
                #         #
                #         # if to_check_with is not None:
                #         #     exists = [to_check_with.c3d[point[0], point[1], point[2]] for point in coord]
                #         #     temp_ind = np.where(exists)[0]
                #         #     coord = np.delete(coord, temp_ind, 0)
                #         #     seeds = np.delete(seeds, temp_ind, 0)
                #
                #         coord = coord.transpose()
                #         seeds = seeds.transpose()
                #
                #         self.cur_case.active.c3d[coord[0], coord[1], coord[2]] -= 1
                #         self.cur_case.product.c3d[coord[0], coord[1], coord[2]] += 1
                #         self.cur_case.product.fix_full_cells(coord)
                #
                #         self.cur_case.oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1
                #         self.cur_case.oxidant.c3d[coord_in[0], coord_in[1], coord_in[2]] -= 1

def ci_single_no_growth_only_p0(cur_case, seeds, oxidant, full_3d):
    shm_p = shared_memory.SharedMemory(name=cur_case.product_c3d_shm_mdata.name)
    product = np.ndarray(cur_case.product_c3d_shm_mdata.shape, dtype=cur_case.product_c3d_shm_mdata.dtype,
                         buffer=shm_p.buf)
    shm_a = shared_memory.SharedMemory(name=cur_case.active_c3d_shm_mdata.name)
    active = np.ndarray(cur_case.active_c3d_shm_mdata.shape, dtype=cur_case.active_c3d_shm_mdata.dtype,
                        buffer=shm_a.buf)
    shm_product_x_nzs = shared_memory.SharedMemory(name=cur_case.prod_indexes_shm_mdata.name)
    product_x_nzs = np.ndarray(cur_case.prod_indexes_shm_mdata.shape, dtype=cur_case.prod_indexes_shm_mdata.dtype,
                               buffer=shm_product_x_nzs.buf)

    all_arounds = calc_sur_ind_formation(seeds, active.shape[2] - 1)
    neighbours = go_around_bool(active, all_arounds[:, :-2])
    arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.short)
    temp_ind = np.where(arr_len_out > 0)[0]

    if len(temp_ind) > 0:
        seeds = seeds[temp_ind]
        neighbours = neighbours[temp_ind]
        all_arounds = all_arounds[temp_ind]
        randomise = np.array(np.random.random_sample(len(seeds)), dtype=np.float64)
        temp_ind = np.where(randomise < cur_case.nucleation_probabilities.nucl_prob.values_pp[0])[0]
        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]
            out_to_del = [np.array(np.nonzero(item)[0]) for item in neighbours]
            to_del = [np.random.choice(item, 1, replace=False) for item in out_to_del]
            coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                             dtype=np.short)
            coord = np.reshape(coord, (len(coord) * 1, 3))
            coord = coord.transpose()
            seeds = seeds.transpose()

            active[coord[0], coord[1], coord[2]] -= 1
            oxidant[seeds[0], seeds[1], seeds[2]] -= 1

            # self.cur_case.product.c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
            product[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

            # self.cur_case.product.fix_full_cells(coord)  # precip on place of active!
            cur_case.fix_full_cells(product, full_3d, seeds, cur_case.oxidation_number)  # precip on place of oxidant!

            # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
            # dissolution function
            product_x_nzs[seeds[2][0]] = True
    shm_p.close()
    shm_a.close()
    shm_product_x_nzs.close()


def ci_single_no_growth(cur_case, seeds, oxidant, full_3d):
    shm_p = shared_memory.SharedMemory(name=cur_case.product_c3d_shm_mdata.name)
    product = np.ndarray(cur_case.product_c3d_shm_mdata.shape, dtype=cur_case.product_c3d_shm_mdata.dtype,
                         buffer=shm_p.buf)
    shm_a = shared_memory.SharedMemory(name=cur_case.active_c3d_shm_mdata.name)
    active = np.ndarray(cur_case.active_c3d_shm_mdata.shape, dtype=cur_case.active_c3d_shm_mdata.dtype,
                        buffer=shm_a.buf)
    shm_product_x_nzs = shared_memory.SharedMemory(name=cur_case.prod_indexes_shm_mdata.name)
    product_x_nzs = np.ndarray(cur_case.prod_indexes_shm_mdata.shape, dtype=cur_case.prod_indexes_shm_mdata.dtype,
                               buffer=shm_product_x_nzs.buf)
    all_arounds = calc_sur_ind_formation(seeds, active.shape[2] - 1)
    neighbours = go_around_bool(active, all_arounds[:, :-2])
    arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.short)
    temp_ind = np.where(arr_len_out > 0)[0]
    if len(temp_ind) > 0:
        seeds = seeds[temp_ind]
        neighbours = neighbours[temp_ind]
        all_arounds = all_arounds[temp_ind]
        out_to_del = [np.array(np.nonzero(item)[0]) for item in neighbours]
        to_del = [np.random.choice(item, 1, replace=False) for item in out_to_del]
        coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                         dtype=np.short)
        coord = np.reshape(coord, (len(coord) * 1, 3))
        coord = coord.transpose()
        seeds = seeds.transpose()
        active[coord[0], coord[1], coord[2]] -= 1
        oxidant[seeds[0], seeds[1], seeds[2]] -= 1
        # self.cur_case.product.c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
        product[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!
        # self.cur_case.product.fix_full_cells(coord)  # precip on place of active!
        cur_case.fix_full_cells(product, full_3d, seeds, cur_case.oxidation_number)   # precip on place of oxidant!
        # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
        # dissolution function
        product_x_nzs[seeds[2][0]] = True


def go_around_mult_oxid_n_also_partial_neigh_aip_MP(array_3d, around_coords):
    return np.sum(go_around_int(array_3d, around_coords), axis=1)
