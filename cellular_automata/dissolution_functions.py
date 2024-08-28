from multiprocessing import shared_memory
from utils.numba_functions import *
from .neigh_indexes import *


def dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL(shm_mdata, chunk_range, comb_ind, aggregated_ind, dissolution_probabilities):
    to_dissolve = np.array([[], [], []], dtype=np.short)
    shm = shared_memory.SharedMemory(name=shm_mdata.name)
    array_3D = np.ndarray(shm_mdata.shape, dtype=shm_mdata.dtype, buffer=shm.buf)
    to_dissol_pn_buffer = np.array([[], [], []], dtype=np.short)

    nz_ind = np.array(np.nonzero(array_3D[chunk_range[0]:chunk_range[1], :, comb_ind]))
    nz_ind[0] += chunk_range[0]
    coord_buffer = nz_ind

    new_data = comb_ind[nz_ind[2]]
    coord_buffer[2, :] = new_data

    if len(coord_buffer[0]) > 0:
        flat_arounds = calc_sur_ind_decompose_flat_with_zero(coord_buffer)

        all_neigh = go_around_int(array_3D, flat_arounds)
        all_neigh[:, 6] -= 1

        all_neigh_block = np.array([])
        all_neigh_no_block = np.array([])
        numb_in_prod_block = np.array([], dtype=int)
        numb_in_prod_no_block = np.array([], dtype=int)

        where_not_null = np.unique(np.where(all_neigh[:, :6] > 0)[0])
        to_dissol_no_neigh = np.array(np.delete(coord_buffer, where_not_null, axis=1), dtype=np.short)
        coord_buffer = coord_buffer[:, where_not_null]

        if len(coord_buffer[0]) > 0:
            all_neigh = all_neigh[where_not_null]
            numb_in_prod = all_neigh[:, -1]

            all_neigh_bool = np.array(all_neigh[:, :6], dtype=bool)

            arr_len_flat = np.sum(all_neigh_bool, axis=1)

            index_outside = np.where((arr_len_flat < 6))[0]
            coord_buffer = coord_buffer[:, index_outside]

            all_neigh_bool = all_neigh_bool[index_outside]
            arr_len_flat = arr_len_flat[index_outside]
            numb_in_prod = numb_in_prod[index_outside]

            non_flat_arounds = calc_sur_ind_decompose_no_flat(coord_buffer)
            non_flat_neigh = go_around_bool(array_3D, non_flat_arounds)
            all_neigh_bool = np.concatenate((all_neigh_bool, non_flat_neigh), axis=1)
            ind_where_blocks = aggregate(aggregated_ind, all_neigh_bool)

            if len(ind_where_blocks) > 0:
                to_dissol_pn_buffer = np.array(np.delete(coord_buffer, ind_where_blocks, axis=1), dtype=np.short)

                all_neigh_no_block = np.delete(arr_len_flat, ind_where_blocks)
                numb_in_prod_no_block = np.delete(numb_in_prod, ind_where_blocks, axis=0)
                coord_buffer = coord_buffer[:, ind_where_blocks]
                all_neigh_block = arr_len_flat[ind_where_blocks]

                numb_in_prod_block = numb_in_prod[ind_where_blocks]
            else:
                to_dissol_pn_buffer = coord_buffer
                all_neigh_no_block = arr_len_flat
                numb_in_prod_no_block = numb_in_prod

                coord_buffer = np.array([[], [], []], dtype=np.ushort)
                all_neigh_block = np.array([])
                numb_in_prod_block = np.array([], dtype=int)

        to_dissolve_no_block = to_dissol_pn_buffer
        probs_no_block = dissolution_probabilities.get_probabilities(all_neigh_no_block, to_dissolve_no_block[2])

        non_z_ind = np.where(numb_in_prod_no_block != 0)[0]
        repeated_coords = np.repeat(to_dissolve_no_block[:, non_z_ind], numb_in_prod_no_block[non_z_ind], axis=1)
        repeated_probs = np.repeat(probs_no_block[non_z_ind], numb_in_prod_no_block[non_z_ind])
        to_dissolve_no_block = np.concatenate((to_dissolve_no_block, repeated_coords), axis=1)
        probs_no_block = np.concatenate((probs_no_block, repeated_probs))
        randomise = np.random.random_sample(len(to_dissolve_no_block[0]))
        temp_ind = np.where(randomise < probs_no_block)[0]
        to_dissolve_no_block = to_dissolve_no_block[:, temp_ind]

        to_dissolve_block = coord_buffer
        probs_block = dissolution_probabilities.get_probabilities_block(all_neigh_block, to_dissolve_block[2])

        non_z_ind = np.where(numb_in_prod_block != 0)[0]
        repeated_coords = np.repeat(to_dissolve_block[:, non_z_ind], numb_in_prod_block[non_z_ind], axis=1)
        repeated_probs = np.repeat(probs_block[non_z_ind], numb_in_prod_block[non_z_ind])
        to_dissolve_block = np.concatenate((to_dissolve_block, repeated_coords), axis=1)
        probs_block = np.concatenate((probs_block, repeated_probs))
        randomise = np.random.random_sample(len(to_dissolve_block[0]))
        temp_ind = np.where(randomise < probs_block)[0]
        to_dissolve_block = to_dissolve_block[:, temp_ind]

        probs_no_neigh = dissolution_probabilities.dissol_prob.values_pp[to_dissol_no_neigh[2]]
        randomise = np.random.random_sample(len(to_dissol_no_neigh[0]))
        temp_ind = np.where(randomise < probs_no_neigh)[0]
        to_dissol_no_neigh = to_dissol_no_neigh[:, temp_ind]

        to_dissolve = np.concatenate((to_dissolve_no_block, to_dissol_no_neigh, to_dissolve_block), axis=1)

    shm.close()
    return to_dissolve


def dissolution_zhou_wei_no_bsf(shm_mdata, chunk_range, comb_ind, aggregated_ind, dissolution_probabilities):
    to_dissolve = np.array([[], [], []], dtype=np.short)
    shm = shared_memory.SharedMemory(name=shm_mdata.name)
    array_3D = np.ndarray(shm_mdata.shape, dtype=shm_mdata.dtype, buffer=shm.buf)
    to_dissol_pn_buffer = np.array([[], [], []], dtype=np.short)

    nz_ind = np.array(np.nonzero(array_3D[chunk_range[0]:chunk_range[1], :, comb_ind]))
    nz_ind[0] += chunk_range[0]
    coord_buffer = nz_ind

    new_data = comb_ind[nz_ind[2]]
    coord_buffer[2, :] = new_data

    if len(coord_buffer[0]) > 0:
        flat_arounds = calc_sur_ind_decompose_flat_with_zero(coord_buffer)

        all_neigh = go_around_int(array_3D, flat_arounds)
        all_neigh[:, 6] -= 1

        # all_neigh_block = np.array([])
        all_neigh_no_block = np.array([])
        # numb_in_prod_block = np.array([], dtype=int)
        numb_in_prod_no_block = np.array([], dtype=int)

        where_not_null = np.unique(np.where(all_neigh[:, :6] > 0)[0])
        to_dissol_no_neigh = np.array(np.delete(coord_buffer, where_not_null, axis=1), dtype=np.short)
        coord_buffer = coord_buffer[:, where_not_null]

        if len(coord_buffer[0]) > 0:
            all_neigh = all_neigh[where_not_null]
            numb_in_prod = all_neigh[:, -1]

            all_neigh_bool = np.array(all_neigh[:, :6], dtype=bool)

            arr_len_flat = np.sum(all_neigh_bool, axis=1)

            index_outside = np.where((arr_len_flat < 6))[0]
            coord_buffer = coord_buffer[:, index_outside]

            # all_neigh_bool = all_neigh_bool[index_outside]
            arr_len_flat = arr_len_flat[index_outside]
            numb_in_prod = numb_in_prod[index_outside]

            # non_flat_arounds = calc_sur_ind_decompose_no_flat(coord_buffer)
            # non_flat_neigh = go_around_bool(array_3D, non_flat_arounds)
            # all_neigh_bool = np.concatenate((all_neigh_bool, non_flat_neigh), axis=1)

            # ind_where_blocks = aggregate(aggregated_ind, all_neigh_bool)

            # if len(ind_where_blocks) > 0:
            #     to_dissol_pn_buffer = np.array(np.delete(coord_buffer, ind_where_blocks, axis=1), dtype=np.short)
            #
            #     all_neigh_no_block = np.delete(arr_len_flat, ind_where_blocks)
            #     numb_in_prod_no_block = np.delete(numb_in_prod, ind_where_blocks, axis=0)
            #     coord_buffer = coord_buffer[:, ind_where_blocks]
            #     all_neigh_block = arr_len_flat[ind_where_blocks]
            #
            #     numb_in_prod_block = numb_in_prod[ind_where_blocks]

            to_dissol_pn_buffer = coord_buffer
            all_neigh_no_block = arr_len_flat
            numb_in_prod_no_block = numb_in_prod

            # coord_buffer = np.array([[], [], []], dtype=np.ushort)
            # all_neigh_block = np.array([])
            # numb_in_prod_block = np.array([], dtype=int)

        to_dissolve_no_block = to_dissol_pn_buffer
        probs_no_block = dissolution_probabilities.get_probabilities(all_neigh_no_block, to_dissolve_no_block[2])

        non_z_ind = np.where(numb_in_prod_no_block != 0)[0]
        repeated_coords = np.repeat(to_dissolve_no_block[:, non_z_ind], numb_in_prod_no_block[non_z_ind], axis=1)
        repeated_probs = np.repeat(probs_no_block[non_z_ind], numb_in_prod_no_block[non_z_ind])
        to_dissolve_no_block = np.concatenate((to_dissolve_no_block, repeated_coords), axis=1)
        probs_no_block = np.concatenate((probs_no_block, repeated_probs))
        randomise = np.random.random_sample(len(to_dissolve_no_block[0]))
        temp_ind = np.where(randomise < probs_no_block)[0]
        to_dissolve_no_block = to_dissolve_no_block[:, temp_ind]

        # to_dissolve_block = coord_buffer
        # probs_block = dissolution_probabilities.get_probabilities_block(all_neigh_block, to_dissolve_block[2])

        # non_z_ind = np.where(numb_in_prod_block != 0)[0]
        # repeated_coords = np.repeat(to_dissolve_block[:, non_z_ind], numb_in_prod_block[non_z_ind], axis=1)
        # repeated_probs = np.repeat(probs_block[non_z_ind], numb_in_prod_block[non_z_ind])
        # to_dissolve_block = np.concatenate((to_dissolve_block, repeated_coords), axis=1)
        # probs_block = np.concatenate((probs_block, repeated_probs))
        # randomise = np.random.random_sample(len(to_dissolve_block[0]))
        # temp_ind = np.where(randomise < probs_block)[0]
        # to_dissolve_block = to_dissolve_block[:, temp_ind]

        probs_no_neigh = dissolution_probabilities.dissol_prob.values_pp[to_dissol_no_neigh[2]]
        randomise = np.random.random_sample(len(to_dissol_no_neigh[0]))
        temp_ind = np.where(randomise < probs_no_neigh)[0]
        to_dissol_no_neigh = to_dissol_no_neigh[:, temp_ind]

        # to_dissolve = np.concatenate((to_dissolve_no_block, to_dissol_no_neigh, to_dissolve_block), axis=1)
        to_dissolve = np.concatenate((to_dissolve_no_block, to_dissol_no_neigh), axis=1)

    shm.close()
    return to_dissolve


def dissolution_zhou_wei_original(shm_mdata, chunk_range, comb_ind, aggregated_ind, dissolution_probabilities):
    to_dissolve = np.array([[], [], []], dtype=np.short)
    shm = shared_memory.SharedMemory(name=shm_mdata.name)
    array_3D = np.ndarray(shm_mdata.shape, dtype=shm_mdata.dtype, buffer=shm.buf)
    to_dissol_pn_buffer = np.array([[], [], []], dtype=np.short)

    nz_ind = np.array(np.nonzero(array_3D[chunk_range[0]:chunk_range[1], :, comb_ind]))
    nz_ind[0] += chunk_range[0]
    coord_buffer = nz_ind

    new_data = comb_ind[nz_ind[2]]
    coord_buffer[2, :] = new_data

    if len(coord_buffer[0]) > 0:
        flat_arounds = calc_sur_ind_decompose_flat_with_zero(coord_buffer)

        all_neigh = go_around_int(array_3D, flat_arounds)
        all_neigh[:, 6] -= 1

        all_neigh_block = np.array([])
        all_neigh_no_block = np.array([])
        numb_in_prod_block = np.array([], dtype=int)
        numb_in_prod_no_block = np.array([], dtype=int)

        where_not_null = np.unique(np.where(all_neigh[:, :6] > 0)[0])
        to_dissol_no_neigh = np.array(np.delete(coord_buffer, where_not_null, axis=1), dtype=np.short)
        coord_buffer = coord_buffer[:, where_not_null]

        if len(coord_buffer[0]) > 0:
            all_neigh = all_neigh[where_not_null]
            numb_in_prod = all_neigh[:, -1]

            all_neigh_bool = np.array(all_neigh[:, :6], dtype=bool)

            arr_len_flat = np.sum(all_neigh_bool, axis=1)

            index_outside = np.where((arr_len_flat < 6))[0]
            coord_buffer = coord_buffer[:, index_outside]

            all_neigh_bool = all_neigh_bool[index_outside]
            arr_len_flat = arr_len_flat[index_outside]
            numb_in_prod = numb_in_prod[index_outside]

            non_flat_arounds = calc_sur_ind_decompose_no_flat(coord_buffer)
            non_flat_neigh = go_around_bool(array_3D, non_flat_arounds)
            all_neigh_bool = np.concatenate((all_neigh_bool, non_flat_neigh), axis=1)
            ind_where_blocks = aggregate(aggregated_ind, all_neigh_bool)

            if len(ind_where_blocks) > 0:
                to_dissol_pn_buffer = np.array(np.delete(coord_buffer, ind_where_blocks, axis=1), dtype=np.short)

                all_neigh_no_block = np.delete(arr_len_flat, ind_where_blocks)
                numb_in_prod_no_block = np.delete(numb_in_prod, ind_where_blocks, axis=0)
                coord_buffer = coord_buffer[:, ind_where_blocks]
                all_neigh_block = arr_len_flat[ind_where_blocks]

                numb_in_prod_block = numb_in_prod[ind_where_blocks]
            else:
                to_dissol_pn_buffer = coord_buffer
                all_neigh_no_block = arr_len_flat
                numb_in_prod_no_block = numb_in_prod

                coord_buffer = np.array([[], [], []], dtype=np.ushort)
                all_neigh_block = np.array([])
                numb_in_prod_block = np.array([], dtype=int)

        to_dissolve_no_block = to_dissol_pn_buffer
        probs_no_block = dissolution_probabilities.dissol_prob.values_pp[to_dissolve_no_block[2]]

        non_z_ind = np.where(numb_in_prod_no_block != 0)[0]
        repeated_coords = np.repeat(to_dissolve_no_block[:, non_z_ind], numb_in_prod_no_block[non_z_ind], axis=1)
        repeated_probs = np.repeat(probs_no_block[non_z_ind], numb_in_prod_no_block[non_z_ind])
        to_dissolve_no_block = np.concatenate((to_dissolve_no_block, repeated_coords), axis=1)
        probs_no_block = np.concatenate((probs_no_block, repeated_probs))
        randomise = np.random.random_sample(len(to_dissolve_no_block[0]))
        temp_ind = np.where(randomise < probs_no_block)[0]
        to_dissolve_no_block = to_dissolve_no_block[:, temp_ind]

        to_dissolve_block = coord_buffer
        probs_block = dissolution_probabilities.p1.values_pp[to_dissolve_block[2]]

        non_z_ind = np.where(numb_in_prod_block != 0)[0]
        repeated_coords = np.repeat(to_dissolve_block[:, non_z_ind], numb_in_prod_block[non_z_ind], axis=1)
        repeated_probs = np.repeat(probs_block[non_z_ind], numb_in_prod_block[non_z_ind])
        to_dissolve_block = np.concatenate((to_dissolve_block, repeated_coords), axis=1)
        probs_block = np.concatenate((probs_block, repeated_probs))
        randomise = np.random.random_sample(len(to_dissolve_block[0]))
        temp_ind = np.where(randomise < probs_block)[0]
        to_dissolve_block = to_dissolve_block[:, temp_ind]

        probs_no_neigh = dissolution_probabilities.dissol_prob.values_pp[to_dissol_no_neigh[2]]
        randomise = np.random.random_sample(len(to_dissol_no_neigh[0]))
        temp_ind = np.where(randomise < probs_no_neigh)[0]
        to_dissol_no_neigh = to_dissol_no_neigh[:, temp_ind]

        to_dissolve = np.concatenate((to_dissolve_no_block, to_dissol_no_neigh, to_dissolve_block), axis=1)

    shm.close()
    return to_dissolve