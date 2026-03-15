from multiprocessing import shared_memory
import numpy as np
from utils.numba_functions import (
    aggregate,
    dissolution_subblock_kernel,
    dissolution_subblock_kernel_snapshot,
    dissolution_subblock_kernel_snapshot_with_blocks,
    go_around_bool,
    go_around_int,
    insert_counts,
)
from .neigh_indexes import (
    ind_decompose_flat_z,
    ind_decompose_no_flat,
    OFFSETS_26,
    calc_sur_ind_decompose_flat_with_zero,
    calc_sur_ind_decompose_no_flat,
)


# ---------------------------------------------------------------------------
# Block patterns: replace aggregated_ind with a single 2D array (n_patterns, 7)
# of indices into the 26 neighbour bools (6 flat + 20 non-flat). Used to detect
# "block" cells when extending dissolution to BSF.
# ---------------------------------------------------------------------------
def get_block_patterns_from_aggregated(aggregated_ind):
    """Convert legacy aggregated_ind (list of index arrays) to (n_patterns, 7) int8 array."""
    if aggregated_ind is None or len(aggregated_ind) == 0:
        return np.zeros((0, 7), dtype=np.int8)
    return np.asarray(aggregated_ind, dtype=np.int8)


def dissolution_subblock_worker(task):
    """
    Worker for x-partitioned dissolution. Each worker processes product cells in its
    x-planes (plane_indexes). Reads neighbour product from any x (read-only); writes
    only product, full_3d, active in its planes. Returns (3, n) to_dissolve coords
    for main process to add oxidant.
    task: (cur_case_mp, plane_indexes, values_pp, const_a_pp, const_b_pp, const_c_pp, const_d_pp).
    Reuses same SHM layout as nucleation (product, full_3d, active).
    """
    (
        cur_case_mp,
        plane_indexes,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
    ) = task
    plane_indexes = np.asarray(plane_indexes, dtype=np.intp).ravel()

    shm_p = shared_memory.SharedMemory(name=cur_case_mp.product_c3d_shm_mdata.name)
    product = np.ndarray(
        cur_case_mp.product_c3d_shm_mdata.shape,
        dtype=cur_case_mp.product_c3d_shm_mdata.dtype,
        buffer=shm_p.buf,
    )
    shm_full = shared_memory.SharedMemory(name=cur_case_mp.full_shm_mdata.name)
    full_3d = np.ndarray(
        cur_case_mp.full_shm_mdata.shape,
        dtype=cur_case_mp.full_shm_mdata.dtype,
        buffer=shm_full.buf,
    )
    shm_a = shared_memory.SharedMemory(name=cur_case_mp.active_c3d_shm_mdata.name)
    active = np.ndarray(
        cur_case_mp.active_c3d_shm_mdata.shape,
        dtype=cur_case_mp.active_c3d_shm_mdata.dtype,
        buffer=shm_a.buf,
    )

    n_i, n_j, n_z = product.shape
    n_cells = n_i
    # Face-only offsets (6), same order as 6 first rows of ind_decompose_flat_z
    face_offsets_6 = np.asarray(ind_decompose_flat_z[:6], dtype=np.int8)

    # Max possible dissolved = total product count in this chunk
    n_plane = plane_indexes.shape[0]
    max_dissolve = int(np.sum(product[plane_indexes, :, :]))
    max_dissolve = max(max_dissolve, 1)

    to_dissolve_out = np.zeros((3, max_dissolve), dtype=np.int32)
    def _extend_to_nz(arr, nz):
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr) >= nz:
            return arr
        out = np.empty(nz, dtype=np.float64)
        out[: len(arr)] = arr
        out[len(arr) :] = arr[-1]
        return out

    values_pp = _extend_to_nz(values_pp, n_z)
    const_a_pp = _extend_to_nz(const_a_pp, n_z)
    const_b_pp = _extend_to_nz(const_b_pp, n_z)
    const_c_pp = _extend_to_nz(const_c_pp, n_z)
    const_d_pp = _extend_to_nz(const_d_pp, n_z)
    seed = np.random.randint(0, 2**31)

    n_dissolved = dissolution_subblock_kernel(
        product,
        full_3d,
        active,
        plane_indexes,
        face_offsets_6,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        n_cells,
        n_z,
        seed,
        to_dissolve_out,
    )

    shm_p.close()
    shm_full.close()
    shm_a.close()

    if n_dissolved == 0:
        return np.array([[], [], []], dtype=np.int32)
    return to_dissolve_out[:, :n_dissolved].copy()


# ---------------------------------------------------------------------------
# V2: Snapshot + workers write directly to oxidant/active SHM (no flat arrays).
# ---------------------------------------------------------------------------


def _views_from_segment_dissol(shm, n, max_per_cell):
    """Same layout as diffusion: [count (n³ int8)][dirs (n³×max_per_cell uint8)]. Returns (count, dirs) flat views."""
    n3 = n * n * n
    count_bytes = n3 * np.dtype(np.int8).itemsize
    dirs_bytes = n3 * max_per_cell * np.dtype(np.uint8).itemsize
    count = np.ndarray((n3,), dtype=np.int8, buffer=shm.buf, offset=0)
    dirs = np.ndarray((n3, max_per_cell), dtype=np.uint8, buffer=shm.buf, offset=count_bytes)
    return count, dirs


def dissolution_subblock_worker_v2(task):
    """
    Worker for x-partitioned dissolution V2. Reads neighbour product from product_snapshot only.
    Writes product, full_3d. Inward (oxidant) and outward (active) particles are written
    inside the dissolution kernel; no to_dissolve_out, no add_dissolution_particles_to_grid.
    """
    (
        cur_case_mp,
        plane_indexes,
        k_lo,
        k_hi,
        values_pp,
        const_a_pp,
        const_b_pp,
        const_c_pp,
        const_d_pp,
        product_snapshot_shm_mdata,
        oxidant_write_shm_mdata,
        max_per_cell_oxidant,
        max_per_cell_active,
        packed_dirs_oxidant,
        block_patterns,
        bsf,
    ) = task
    plane_indexes = np.asarray(plane_indexes, dtype=np.intp).ravel()
    k_lo = int(k_lo)
    k_hi = int(k_hi)
    threshold_inward = int(getattr(cur_case_mp, "threshold_inward", 1))
    threshold_outward = int(getattr(cur_case_mp, "threshold_outward", 1))
    dissolution_thresholds = np.array([threshold_inward, threshold_outward], dtype=np.int32)

    shm_snap = shared_memory.SharedMemory(name=product_snapshot_shm_mdata.name)
    product_read = np.ndarray(
        product_snapshot_shm_mdata.shape,
        dtype=product_snapshot_shm_mdata.dtype,
        buffer=shm_snap.buf,
    )
    shm_p = shared_memory.SharedMemory(name=cur_case_mp.product_c3d_shm_mdata.name)
    product = np.ndarray(
        cur_case_mp.product_c3d_shm_mdata.shape,
        dtype=cur_case_mp.product_c3d_shm_mdata.dtype,
        buffer=shm_p.buf,
    )
    shm_full = shared_memory.SharedMemory(name=cur_case_mp.full_shm_mdata.name)
    full_3d = np.ndarray(
        cur_case_mp.full_shm_mdata.shape,
        dtype=cur_case_mp.full_shm_mdata.dtype,
        buffer=shm_full.buf,
    )
    shm_a = shared_memory.SharedMemory(name=cur_case_mp.active_c3d_shm_mdata.name)
    n_i, n_j, n_z = cur_case_mp.active_c3d_shm_mdata.shape
    active_count, active_dirs = _views_from_segment_dissol(shm_a, n_i, max_per_cell_active)
    shm_ox_w = shared_memory.SharedMemory(name=oxidant_write_shm_mdata.name)
    n_ox = oxidant_write_shm_mdata.shape[0]
    oxidant_count, oxidant_dirs = _views_from_segment_dissol(shm_ox_w, n_ox, max_per_cell_oxidant)
    packed_dirs = np.asarray(packed_dirs_oxidant, dtype=np.uint8).ravel()

    n_cells = n_i
    offsets_26 = np.asarray(OFFSETS_26, dtype=np.int8)

    def _extend_to_nz(arr, nz):
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr) >= nz:
            return arr
        out = np.empty(nz, dtype=np.float64)
        out[: len(arr)] = arr
        out[len(arr) :] = arr[-1]
        return out

    values_pp = _extend_to_nz(values_pp, n_z)
    const_a_pp = _extend_to_nz(const_a_pp, n_z)
    const_b_pp = _extend_to_nz(const_b_pp, n_z)
    const_c_pp = _extend_to_nz(const_c_pp, n_z)
    const_d_pp = _extend_to_nz(const_d_pp, n_z)
    seed = np.random.randint(0, 2**31)

    use_blocks = (
        block_patterns is not None
        and getattr(block_patterns, "shape", (0,))[0] > 0
        and float(bsf) >= 1.0
    )
    if use_blocks:
        block_pat = np.asarray(block_patterns, dtype=np.int8)
        dissolution_subblock_kernel_snapshot_with_blocks(
            product_read,
            product,
            full_3d,
            oxidant_count,
            oxidant_dirs,
            active_count,
            active_dirs,
            plane_indexes,
            offsets_26,
            k_lo,
            k_hi,
            block_pat,
            float(bsf),
            values_pp,
            const_a_pp,
            const_b_pp,
            const_c_pp,
            const_d_pp,
            n_cells,
            n_z,
            seed,
            dissolution_thresholds,
            max_per_cell_oxidant,
            max_per_cell_active,
            packed_dirs,
        )
    else:
        dissolution_subblock_kernel_snapshot(
            product_read,
            product,
            full_3d,
            oxidant_count,
            oxidant_dirs,
            active_count,
            active_dirs,
            plane_indexes,
            offsets_26,
            k_lo,
            k_hi,
            values_pp,
            const_a_pp,
            const_b_pp,
            const_c_pp,
            const_d_pp,
            n_cells,
            n_z,
            seed,
            dissolution_thresholds,
            max_per_cell_oxidant,
            max_per_cell_active,
            packed_dirs,
        )

    shm_snap.close()
    shm_p.close()
    shm_full.close()
    shm_a.close()
    shm_ox_w.close()
    return None


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