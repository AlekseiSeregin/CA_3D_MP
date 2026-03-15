from utils.numba_functions import *
from utils.numba_functions import nucleation_subblock_kernel, nucleation_subblock_kernel_simple
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


def precip_step_subblock_worker(task):
    """
    One worker for z-subblock nucleation: owns active cells with k in [k_lo, k_hi].
    Seed slab: k in [max(0, k_lo-1), min(n_z-1, k_hi+1)]. For each cell in the seed slab
    with oxidant > 0 and not full, perform up to oxidant[i,j,k] nucleation attempts,
    respecting oxidation_number (product cap) and only using active neighbours in [k_lo, k_hi].
    Inward/outward buffers are diffusion segments [count | dirs]; we decrement count and zero
    the freed dir slot so the segment stays consistent.
    task: (cur_case_mp, k_lo, k_hi, plane_indexes, max_per_cell_oxidant, max_per_cell_active, ind_form).
    plane_indexes = x-axis (i) indices; worker's z range is [k_lo, k_hi].
    """
    cur_case_mp, k_lo, k_hi, plane_indexes, max_per_cell_o, max_per_cell_a, ind_form = task
    plane_indexes = np.asarray(plane_indexes, dtype=np.intp).ravel()
    shm_o = shared_memory.SharedMemory(name=cur_case_mp.oxidant_c3d_shm_mdata.name)
    n_i, n_j, n_z = cur_case_mp.oxidant_c3d_shm_mdata.shape
    n3 = n_i * n_j * n_z
    count_bytes_o = n3 * np.dtype(np.int8).itemsize
    # Diffusion segment: [count (n³ int8)][dirs (n³ × max_per_cell uint8)]; count view F-order
    oxidant = np.ndarray(
        (n_i, n_j, n_z),
        dtype=np.int8,
        buffer=shm_o.buf,
        offset=0,
        order="F",
    )
    oxidant_dirs = np.ndarray(
        (n3, max_per_cell_o),
        dtype=np.uint8,
        buffer=shm_o.buf,
        offset=count_bytes_o,
    )
    shm_a = shared_memory.SharedMemory(name=cur_case_mp.active_c3d_shm_mdata.name)
    count_bytes_a = n3 * np.dtype(np.int8).itemsize
    active = np.ndarray(
        (n_i, n_j, n_z),
        dtype=np.int8,
        buffer=shm_a.buf,
        offset=0,
        order="F",
    )
    active_dirs = np.ndarray(
        (n3, max_per_cell_a),
        dtype=np.uint8,
        buffer=shm_a.buf,
        offset=count_bytes_a,
    )
    shm_p = shared_memory.SharedMemory(name=cur_case_mp.product_c3d_shm_mdata.name)
    product = np.ndarray(cur_case_mp.product_c3d_shm_mdata.shape, dtype=cur_case_mp.product_c3d_shm_mdata.dtype, buffer=shm_p.buf)
    shm_full = shared_memory.SharedMemory(name=cur_case_mp.full_shm_mdata.name)
    full_3d = np.ndarray(cur_case_mp.full_shm_mdata.shape, dtype=cur_case_mp.full_shm_mdata.dtype, buffer=shm_full.buf)
    shm_product_init = shared_memory.SharedMemory(name=cur_case_mp.precip_3d_init_shm_mdata.name)
    product_init = np.ndarray(cur_case_mp.precip_3d_init_shm_mdata.shape, dtype=cur_case_mp.precip_3d_init_shm_mdata.dtype, buffer=shm_product_init.buf)
    shm_product_x_nzs = shared_memory.SharedMemory(name=cur_case_mp.prod_indexes_shm_mdata.name)
    product_x_nzs = np.ndarray(cur_case_mp.prod_indexes_shm_mdata.shape, dtype=cur_case_mp.prod_indexes_shm_mdata.dtype, buffer=shm_product_x_nzs.buf)

    k_seed_lo = max(0, k_lo - 1)
    k_seed_hi = min(full_3d.shape[2] - 1, k_hi + 1)
    # seed_slab_k = z range for this worker (plane_indexes are x indices, not z)
    seed_slab_k = np.arange(k_seed_lo, k_seed_hi + 1, dtype=np.intp)

    ox_num = cur_case_mp.oxidation_number
    nucl_prob = cur_case_mp.nucleation_probabilities
    n_cells = n_i

    # Array 1: offsets for checking actives in yz plane (one step each direction + center), shape (9, 3)
    active_check_offsets = np.asarray(ind_form[:9], dtype=np.int8)

    # Array 2: flat neighbours of cubic cell for product_init count and probability
    # ox_num==1: product only on empty cell → 6 face neighbours (no center)
    # ox_num>1: product can sit oxidation_number times → 6 face + center (0,0,0)
    face_offsets = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.int8,
    )
    if ox_num > 1:
        flat_neigh_offsets = np.vstack([face_offsets, [[0, 0, 0]]]).astype(np.int8)
    else:
        flat_neigh_offsets = face_offsets

    values_pp = np.asarray(nucl_prob.nucl_prob.values_pp, dtype=np.float64)
    const_a_pp = np.asarray(nucl_prob.const_a_pp, dtype=np.float64)
    const_b_pp = np.asarray(nucl_prob.const_b_pp, dtype=np.float64)
    const_c_pp = np.asarray(nucl_prob.const_c_pp, dtype=np.float64)
    const_d_pp = np.asarray(nucl_prob.const_d_pp, dtype=np.float64)
    seed = np.random.randint(0, 2**31)
    # Max x (i) where any oxidant exists (skip x planes with no particles)
    flat = np.flatnonzero(oxidant.ravel(order="F") > 0)
    x_max = int(np.max(flat % n_i)) if flat.size > 0 else -1

    use_simple_nucleation = bool(getattr(cur_case_mp, "use_simple_nucleation", False))
    if use_simple_nucleation:
        nucleation_subblock_kernel_simple(
            oxidant,
            oxidant_dirs,
            active,
            active_dirs,
            product,
            full_3d,
            product_x_nzs,
            ox_num,
            seed_slab_k,
            plane_indexes,
            active_check_offsets,
            n_cells,
            x_max,
            seed,
        )
    else:
        nucleation_subblock_kernel(
            oxidant,
            oxidant_dirs,
            active,
            active_dirs,
            product,
            full_3d,
            product_init,
            product_x_nzs,
            ox_num,
            seed_slab_k,
            plane_indexes,
            active_check_offsets,
            flat_neigh_offsets,
            values_pp,
            const_a_pp,
            const_b_pp,
            const_c_pp,
            const_d_pp,
            n_cells,
            x_max,
            seed,
        )

    shm_o.close()
    shm_a.close()
    shm_p.close()
    shm_full.close()
    shm_product_init.close()
    shm_product_x_nzs.close()


def ci_multi(cur_case, seeds, oxidant, full_3d):
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
    # self_neighbours = np.array(self_neighbours, dtype=bool)
    arr_len_self = np.array([np.sum(item) for item in self_neighbours], dtype=np.short)
    temp_ind = np.where(arr_len_self >= cur_case.threshold_inward - 1)[0]

    if len(temp_ind) > 0:
        seeds = seeds[temp_ind]
        self_neighbours = self_neighbours[temp_ind]
        all_arounds = all_arounds[temp_ind]

        neighbours = go_around_int(active, all_arounds[:, :-2])
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

                out_to_del = [np.array(np.nonzero(item)[0]) for item in neighbours]
                n_rep = [neighbours[ind][pos] for ind, pos in enumerate(out_to_del)]
                corr_arounds = [all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(out_to_del)]
                repeated_coords = [np.repeat(arounds, n_r, axis=0) for arounds, n_r in zip(corr_arounds, n_rep)]
                ind_to_choose = [np.random.choice(len(coord), cur_case.threshold_outward, replace=False) for coord in repeated_coords]
                out_coord = np.array([repeated_coords[ind][pos] for ind, pos in enumerate(ind_to_choose)], dtype=np.short)

                in_to_del = [np.array(np.nonzero(item)[0]) for item in self_neighbours]
                n_rep = [self_neighbours[ind][pos] for ind, pos in enumerate(in_to_del)]
                corr_arounds = [all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(in_to_del)]
                repeated_coords = [np.repeat(arounds, n_r, axis=0) for arounds, n_r in zip(corr_arounds, n_rep)]
                ind_to_choose = [np.random.choice(len(coord), cur_case.threshold_inward - 1, replace=False) for coord in
                                 repeated_coords]
                in_coord = np.array([repeated_coords[ind][pos] for ind, pos in enumerate(ind_to_choose)], dtype=np.short)

                out_coord = np.reshape(out_coord, (len(out_coord) * cur_case.threshold_outward, 3))
                in_coord = np.reshape(in_coord, (len(in_coord) * (cur_case.threshold_inward - 1), 3))
                out_coord = out_coord.transpose()
                in_coord = in_coord.transpose()
                seeds = seeds.transpose()

                # active[out_coord[0], out_coord[1], out_coord[2]] -= 1
                just_decrease_counts(active, out_coord)
                # oxidant[in_coord[0], in_coord[1], in_coord[2]] -= 1
                just_decrease_counts(oxidant, in_coord)
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


def ci_multi_no_active(cur_case, seeds, oxidant, full_3d):
    shm_p = shared_memory.SharedMemory(name=cur_case.product_c3d_shm_mdata.name)
    product = np.ndarray(cur_case.product_c3d_shm_mdata.shape, dtype=cur_case.product_c3d_shm_mdata.dtype,
                         buffer=shm_p.buf)
    shm_product_init = shared_memory.SharedMemory(name=cur_case.precip_3d_init_shm_mdata.name)
    product_init = np.ndarray(cur_case.precip_3d_init_shm_mdata.shape, dtype=cur_case.precip_3d_init_shm_mdata.dtype,
                              buffer=shm_product_init.buf)
    shm_product_x_nzs = shared_memory.SharedMemory(name=cur_case.prod_indexes_shm_mdata.name)
    product_x_nzs = np.ndarray(cur_case.prod_indexes_shm_mdata.shape, dtype=cur_case.prod_indexes_shm_mdata.dtype,
                               buffer=shm_product_x_nzs.buf)

    all_arounds = calc_sur_ind_formation(seeds, cur_case.active_c3d_shm_mdata.shape[2] - 1)

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
        seeds = seeds.transpose()
        product[seeds[0], seeds[1], seeds[2]] += 1
        cur_case.fix_full_cells(product, full_3d, seeds, cur_case.oxidation_number)

        # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
        # dissolution function
        product_x_nzs[seeds[2][0]] = True

    shm_p.close()
    shm_product_init.close()
    shm_product_x_nzs.close()


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


def ci_multi_no_growth(cur_case, seeds, oxidant, full_3d):

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
    self_neighbours = go_around_int(oxidant, all_arounds[:, :-2])
    self_neighbours[:, 4] -= 1
    arr_len_self = np.array([np.sum(item) for item in self_neighbours], dtype=np.short)
    temp_ind = np.where(arr_len_self >= cur_case.threshold_inward - 1)[0]

    if len(temp_ind) > 0:
        seeds = seeds[temp_ind]
        self_neighbours = self_neighbours[temp_ind]
        all_arounds = all_arounds[temp_ind]

        neighbours = go_around_int(active, all_arounds[:, :-2])
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.short)
        temp_ind = np.where(arr_len_out >= cur_case.threshold_outward)[0]

        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            self_neighbours = self_neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]

            out_to_del = [np.array(np.nonzero(item)[0]) for item in neighbours]
            n_rep = [neighbours[ind][pos] for ind, pos in enumerate(out_to_del)]
            corr_arounds = [all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(out_to_del)]
            repeated_coords = [np.repeat(arounds, n_r, axis=0) for arounds, n_r in zip(corr_arounds, n_rep)]
            ind_to_choose = [np.random.choice(len(coord), cur_case.threshold_outward, replace=False) for coord in repeated_coords]
            out_coord = np.array([repeated_coords[ind][pos] for ind, pos in enumerate(ind_to_choose)], dtype=np.short)

            in_to_del = [np.array(np.nonzero(item)[0]) for item in self_neighbours]
            n_rep = [self_neighbours[ind][pos] for ind, pos in enumerate(in_to_del)]
            corr_arounds = [all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(in_to_del)]
            repeated_coords = [np.repeat(arounds, n_r, axis=0) for arounds, n_r in zip(corr_arounds, n_rep)]
            ind_to_choose = [np.random.choice(len(coord), cur_case.threshold_inward - 1, replace=False) for coord in
                             repeated_coords]

            in_coord = np.array([repeated_coords[ind][pos] for ind, pos in enumerate(ind_to_choose)], dtype=np.short)
            out_coord = np.reshape(out_coord, (len(out_coord) * cur_case.threshold_outward, 3))
            in_coord = np.reshape(in_coord, (len(in_coord) * (cur_case.threshold_inward - 1), 3))
            out_coord = out_coord.transpose()
            in_coord = in_coord.transpose()
            seeds = seeds.transpose()

            just_decrease_counts(active, out_coord)
            just_decrease_counts(oxidant, in_coord)
            oxidant[seeds[0], seeds[1], seeds[2]] -= 1
            product[seeds[0], seeds[1], seeds[2]] += 1

            cur_case.fix_full_cells(product, full_3d, seeds, cur_case.oxidation_number)

            # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
            # dissolution function
            product_x_nzs[seeds[2][0]] = True
    shm_p.close()
    shm_a.close()
    shm_product_x_nzs.close()

def go_around_mult_oxid_n_also_partial_neigh_aip_MP(array_3d, around_coords):
    return np.sum(go_around_int(array_3d, around_coords), axis=1)

def go_around_mult_oxid_n_BOOl(array_3d, around_coords):
    return np.sum(go_around_bool(array_3d, around_coords), axis=1)
