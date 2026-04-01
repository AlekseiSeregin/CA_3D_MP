import numba
import numpy as np
# from scipy.special import dtype

# Cache is True by default, but we can set it to False to force Numba to recompile the function on each call
_CACHE = False


@numba.njit(fastmath=True, cache=_CACHE)
def product_counts_upto_bound_from_state(owner_phase, state_count, phase_id, u_bound):
    """Per-x-page product counts from unified state for k in [0, u_bound]."""
    pid = np.uint8(phase_id)
    ub = int(u_bound)
    out = np.zeros(ub + 1, dtype=np.uint32)
    n_i = owner_phase.shape[0]
    n_j = owner_phase.shape[1]
    for k in range(ub + 1):
        s = 0
        for i in range(n_i):
            for j in range(n_j):
                if owner_phase[i, j, k] == pid:
                    s += int(state_count[i, j, k])
        out[k] = s
    return out


@numba.njit(fastmath=True, cache=_CACHE)
def product_counts_at_indexes_from_state(owner_phase, state_count, phase_id, page_indexes):
    """Per-x-page product counts from unified state for explicit page indexes."""
    pid = np.uint8(phase_id)
    n_i = owner_phase.shape[0]
    n_j = owner_phase.shape[1]
    out = np.zeros(page_indexes.shape[0], dtype=np.uint32)
    for p in range(page_indexes.shape[0]):
        k = int(page_indexes[p])
        s = 0
        for i in range(n_i):
            for j in range(n_j):
                if owner_phase[i, j, k] == pid:
                    s += int(state_count[i, j, k])
        out[p] = s
    return out


@numba.njit(fastmath=True, cache=_CACHE)
def init_particles_rand(count, dirs, n, n2, max_per_cell, total_particles, packed_dirs, i_lo, seed):
    """Place total_particles randomly in 3D grid (x >= i_lo when half fill); cap at max_per_cell per cell. Fills count and dirs in place."""
    np.random.seed(seed)
    placed = 0
    while placed < total_particles:
        i = np.random.randint(i_lo, n)
        j = np.random.randint(0, n)
        k = np.random.randint(0, n)
        idx = i + n * j + n2 * k
        if count[idx] < max_per_cell:
            c = count[idx]
            dirs[idx, c] = packed_dirs[np.random.randint(0, 6)]
            count[idx] = c + 1
            placed += 1


@numba.njit(fastmath=True, cache=_CACHE)
def init_particles_exact(count, dirs, n, n2, max_per_cell, n_per, i_lo, packed_dirs, seed):
    """Place n_per particles per x-slice (i_lo..n-1), random (y,z) position without replacement per slice. Fills count and dirs in place."""
    np.random.seed(seed)
    nn = n * n
    for i in range(i_lo, n):
        # Fisher–Yates shuffle to get n_per distinct 2D (j,k) indices
        arr = np.arange(nn)
        for ii in range(nn - 1, nn - n_per - 1, -1):
            jj = np.random.randint(0, ii + 1)
            arr[ii], arr[jj] = arr[jj], arr[ii]
        for p in range(n_per):
            idx_2d = arr[nn - 1 - p]
            j = idx_2d % n
            k = idx_2d // n
            idx = i + n * j + n2 * k
            if count[idx] < max_per_cell:
                c = count[idx]
                dirs[idx, c] = packed_dirs[np.random.randint(0, 6)]
                count[idx] = c + 1


@numba.njit(fastmath=True, cache=_CACHE)
def fill_first_page_kernel(count, dirs, n, n2, max_per_cell, j_coords, k_coords, dir_packed):
    """
    For each (j_coords[i], k_coords[i]) on the x=0 plane: if the cell is not full, add one particle
    with direction dir_packed[i]. If full, do nothing. In place. Same layout as diffusion.
    Returns number of particles not inserted.
    """
    not_inserted = 0
    for i in range(len(j_coords)):
        j = j_coords[i]
        k = k_coords[i]
        idx = n * j + n2 * k
        if count[idx] < max_per_cell:
            c = count[idx]
            dirs[idx, c] = dir_packed[i]
            count[idx] = count[idx] + 1
        else:
            not_inserted += 1
    return not_inserted


@numba.njit(fastmath=True, cache=_CACHE)
def fill_last_page_kernel(count, dirs, n, n2, max_per_cell, j_coords, k_coords, dir_packed):
    """
    For each (j_coords[i], k_coords[i]) on the x=n-1 plane: if the cell is not full, add one particle
    with direction dir_packed[i]. F-order index for (n-1, j, k) is (n-1) + n*j + n2*k.
    Returns number of particles not inserted.
    """
    not_inserted = 0
    x_last = n - 1
    for i in range(len(j_coords)):
        j = j_coords[i]
        k = k_coords[i]
        idx = x_last + n * j + n2 * k
        if count[idx] < max_per_cell:
            c = count[idx]
            dirs[idx, c] = dir_packed[i]
            count[idx] = count[idx] + 1
        else:
            not_inserted += 1
    return not_inserted


@numba.njit(fastmath=True, cache=_CACHE)
def add_dissolution_particles_to_grid(
    oxidant_count,
    oxidant_dirs,
    active_count,
    active_dirs,
    to_dissolve,
    dissolution_thresholds,
    max_per_cell_ox,
    max_per_cell_active,
    n,
    packed_dirs,
    seed,
):
    """
    Add both inward (oxidant) and outward (active) particles from dissolution.
    dissolution_thresholds[0]=inward, [1]=outward. For each (i,j,k) in to_dissolve, add
    that many oxidant/active particles (count + dirs). F-order flat index: i + n*j + n*n*k.
    """
    threshold_inward = int(dissolution_thresholds[0])
    threshold_outward = int(dissolution_thresholds[1])
    np.random.seed(seed)
    n2 = n * n
    n_packed = packed_dirs.shape[0]
    for col in range(to_dissolve.shape[1]):
        i = int(to_dissolve[0, col])
        j = int(to_dissolve[1, col])
        k = int(to_dissolve[2, col])
        nidx = i + n * j + n2 * k
        for _ in range(threshold_inward):
            slot = int(oxidant_count[nidx])
            if slot < max_per_cell_ox:
                r = np.random.randint(0, n_packed)
                oxidant_dirs[nidx, slot] = packed_dirs[r]
                oxidant_count[nidx] = slot + 1
        for _ in range(threshold_outward):
            slot = int(active_count[nidx])
            if slot < max_per_cell_active:
                r = np.random.randint(0, n_packed)
                active_dirs[nidx, slot] = packed_dirs[r]
                active_count[nidx] = slot + 1


@numba.njit(fastmath=True, cache=_CACHE)
def _nucleation_subblock_apply_pbc(ii, jj, kk, n_cells):
    """x: no wrap (hard boundaries at 0 and n_cells-1). y,z: periodic. Returns (ii, jj, kk, valid)."""
    if ii < 0 or ii >= n_cells:
        return ii, jj, kk, False
    jj = ((jj % n_cells) + n_cells) % n_cells
    kk = ((kk % n_cells) + n_cells) % n_cells
    return ii, jj, kk, True


# @numba.njit(fastmath=True, cache=_CACHE)
# def nucleation_subblock_kernel(
#     oxidant,
#     oxidant_dirs,
#     active,
#     active_dirs,
#     product,
#     full_3d,
#     product_init,
#     product_x_nzs,
#     ox_num,
#     seed_slab_k,
#     plane_indexes,
#     active_check_offsets,
#     flat_neigh_offsets,
#     values_pp,
#     const_a_pp,
#     const_b_pp,
#     const_c_pp,
#     const_d_pp,
#     n_cells,
#     seed,
# ):
#     """
#     Numba kernel for subblock nucleation. active_check_offsets = (9, 3) yz-plane offsets including
#     center for checking actives. flat_neigh_offsets = (6, 3) or (7, 3) face neighbours for
#     product_init count (7 if ox_num > 1, includes center). product/full_3d/product_x_nzs written to SHM.
#     """
#     np.random.seed(seed)
#     n2 = n_cells * n_cells
#     n_active = active_check_offsets.shape[0]

#     for k in seed_slab_k:
#         for i in plane_indexes:
#             for j in range(n_cells):
#                 c_max = int(oxidant[i, j, k])
#                 if full_3d[i, j, k] or c_max <= 0:
#                     continue
#                 for _ in range(c_max):
#                     if product[i, j, k] >= ox_num:
#                         break
#                     valid_count = 0
#                     _ni = np.empty(n_active, dtype=np.intp)
#                     _nj = np.empty(n_active, dtype=np.intp)
#                     _nk = np.empty(n_active, dtype=np.intp)
#                     for row in active_check_offsets:
#                         di, dj, dk = int(row[0]), int(row[1]), int(row[2])
#                         ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
#                             i + di, j + dj, k + dk, n_cells
#                         )
#                         if valid and active[ii, jj, kk] > 0:
#                             _ni[valid_count] = ii
#                             _nj[valid_count] = jj
#                             _nk[valid_count] = kk
#                             valid_count += 1
#                     if valid_count == 0:
#                         continue
#                     flat_count = 0
#                     for row in flat_neigh_offsets:
#                         di, dj, dk = int(row[0]), int(row[1]), int(row[2])
#                         ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
#                             i + di, j + dj, k + dk, n_cells
#                         )
#                         if valid and product_init[ii, jj, kk] > 0:
#                             flat_count += 1
#                     # Probability
#                     if flat_count == 0:
#                         prob = values_pp[k]
#                     else:
#                         prob = (
#                             const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k])
#                             + const_d_pp[k]
#                         )
#                     if np.random.random() >= prob:
#                         continue
#                     # Pick one valid neighbour
#                     pick = np.random.randint(0, valid_count)
#                     ni, nj, nk = _ni[pick], _nj[pick], _nk[pick]
#                     # Decrement counts and zero the freed dir slot (diffusion segment [count|dirs])
#                     idx_a = ni + n_cells * nj + n2 * nk
#                     slot_a = active[ni, nj, nk] - 1
#                     active_dirs[idx_a, slot_a] = 0
#                     active[ni, nj, nk] -= 1
#                     idx_o = i + n_cells * j + n2 * k
#                     slot_o = oxidant[i, j, k] - 1
#                     oxidant_dirs[idx_o, slot_o] = 0
#                     oxidant[i, j, k] -= 1
#                     product[i, j, k] += 1
#                     if product[i, j, k] >= ox_num:
#                         full_3d[i, j, k] = True
#                     product_x_nzs[k] = True


# @numba.njit(fastmath=True, cache=_CACHE)
# def nucleation_subblock_kernel_simple(
#     oxidant,
#     oxidant_dirs,
#     active,
#     active_dirs,
#     product,
#     full_3d,
#     product_x_nzs,
#     ox_num,
#     seed_slab_k,
#     plane_indexes,
#     active_check_offsets,
#     n_cells,
#     seed,
# ):
#     """
#     Simplified nucleation: if an inward particle (oxidant) has at least one outward active
#     neighbour, nucleation happens (pick one at random). No flat-neighbour count, no probabilities.
#     Same in-place updates to oxidant/active/product/full_3d/product_x_nzs.
#     """
#     np.random.seed(seed)
#     n2 = n_cells * n_cells
#     n_active = active_check_offsets.shape[0]

#     for k in seed_slab_k:
#         for i in plane_indexes:
#             for j in range(n_cells):
#                 if full_3d[i, j, k] or oxidant[i, j, k] <= 0:
#                     continue
#                 c_max = int(oxidant[i, j, k])
#                 for _ in range(c_max):
#                     if product[i, j, k] >= ox_num:
#                         break
#                     valid_count = 0
#                     _ni = np.empty(n_active, dtype=np.intp)
#                     _nj = np.empty(n_active, dtype=np.intp)
#                     _nk = np.empty(n_active, dtype=np.intp)
#                     for row in active_check_offsets:
#                         di, dj, dk = int(row[0]), int(row[1]), int(row[2])
#                         ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
#                             i + di, j + dj, k + dk, n_cells
#                         )
#                         if valid and active[ii, jj, kk] > 0:
#                             _ni[valid_count] = ii
#                             _nj[valid_count] = jj
#                             _nk[valid_count] = kk
#                             valid_count += 1
#                     if valid_count == 0:
#                         continue
#                     # No probability: pick one valid neighbour and nucleate
#                     pick = np.random.randint(0, valid_count)
#                     ni, nj, nk = _ni[pick], _nj[pick], _nk[pick]
#                     idx_a = ni + n_cells * nj + n2 * nk
#                     slot_a = active[ni, nj, nk] - 1
#                     active_dirs[idx_a, slot_a] = 0
#                     active[ni, nj, nk] -= 1
#                     idx_o = i + n_cells * j + n2 * k
#                     slot_o = oxidant[i, j, k] - 1
#                     oxidant_dirs[idx_o, slot_o] = 0
#                     oxidant[i, j, k] -= 1
#                     product[i, j, k] += 1
#                     if product[i, j, k] >= ox_num:
#                         full_3d[i, j, k] = True
#                     product_x_nzs[k] = True


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_owner(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product_init,
    product_state,
    phase_id,
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
    seed
):
    """Owner-aware legacy probabilistic nucleation (1 inward : 1 outward)."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    n_active = active_check_offsets.shape[0]
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)

    for k in seed_slab_k:
        for i in plane_indexes:
            for j in range(n_cells):
                owner = owner_phase[i, j, k]
                if owner != 0 and owner != pid:
                    continue
                c_max = int(oxidant[i, j, k])
                if c_max <= 0:
                    continue
                idx_o = i + n_cells * j + n2 * k
                for _ in range(c_max):
                    if int(state_count[i, j, k]) >= ox_num or oxidant[i, j, k] <= 0:
                        break
                    if owner != 0 and owner != pid:
                        break
                    valid_count = 0
                    _ni = np.empty(n_active, dtype=np.intp)
                    _nj = np.empty(n_active, dtype=np.intp)
                    _nk = np.empty(n_active, dtype=np.intp)
                    for row in active_check_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
                            i + di, j + dj, k + dk, n_cells
                        )
                        if valid and active[ii, jj, kk] > 0:
                            _ni[valid_count] = ii
                            _nj[valid_count] = jj
                            _nk[valid_count] = kk
                            valid_count += 1
                    if valid_count == 0:
                        continue

                    flat_count = 0
                    for row in flat_neigh_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
                            i + di, j + dj, k + dk, n_cells
                        )
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = (
                            const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k])
                            + const_d_pp[k]
                        )
                    if np.random.random() >= prob:
                        continue

                    pick = np.random.randint(0, valid_count)
                    ni, nj, nk = _ni[pick], _nj[pick], _nk[pick]
                    idx_a = ni + n_cells * nj + n2 * nk
                    slot_a = active[ni, nj, nk] - 1
                    active_dirs[idx_a, slot_a] = 0
                    active[ni, nj, nk] -= 1

                    slot_o = oxidant[i, j, k] - 1
                    oxidant_dirs[idx_o, slot_o] = 0
                    oxidant[i, j, k] -= 1

                    if owner == 0:
                        owner_phase[i, j, k] = pid
                    state_count[i, j, k] = np.uint8(int(state_count[i, j, k]) + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_simple_owner(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product_init,
    product_state,
    phase_id,
    ox_num,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    n_cells,
    seed,
):
    """Owner-aware legacy simplified nucleation (1 inward : 1 outward, no probability)."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    n_active = active_check_offsets.shape[0]
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)

    for k in seed_slab_k:
        for i in plane_indexes:
            for j in range(n_cells):
                owner = owner_phase[i, j, k]
                if owner != 0 and owner != pid:
                    continue
                c_max = int(oxidant[i, j, k])
                if c_max <= 0:
                    continue
                idx_o = i + n_cells * j + n2 * k
                for _ in range(c_max):
                    if int(state_count[i, j, k]) >= ox_num or oxidant[i, j, k] <= 0:
                        break
                    if owner != 0 and owner != pid:
                        break
                    valid_count = 0
                    _ni = np.empty(n_active, dtype=np.intp)
                    _nj = np.empty(n_active, dtype=np.intp)
                    _nk = np.empty(n_active, dtype=np.intp)
                    for row in active_check_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
                            i + di, j + dj, k + dk, n_cells
                        )
                        if valid and active[ii, jj, kk] > 0:
                            _ni[valid_count] = ii
                            _nj[valid_count] = jj
                            _nk[valid_count] = kk
                            valid_count += 1
                    if valid_count == 0:
                        continue

                    pick = np.random.randint(0, valid_count)
                    ni, nj, nk = _ni[pick], _nj[pick], _nk[pick]
                    idx_a = ni + n_cells * nj + n2 * nk
                    slot_a = active[ni, nj, nk] - 1
                    active_dirs[idx_a, slot_a] = 0
                    active[ni, nj, nk] -= 1

                    slot_o = oxidant[i, j, k] - 1
                    oxidant_dirs[idx_o, slot_o] = 0
                    oxidant[i, j, k] -= 1

                    if owner == 0:
                        owner_phase[i, j, k] = pid
                    state_count[i, j, k] = np.uint8(int(state_count[i, j, k]) + 1)


# @numba.njit(fastmath=True, cache=_CACHE)
# def nucleation_subblock_kernel_stoich(
#     oxidant,
#     oxidant_dirs,
#     active,
#     active_dirs,
#     product_init,
#     ox_num,
#     threshold_inward,
#     threshold_outward,
#     seed_slab_k,
#     plane_indexes,
#     active_check_offsets,
#     flat_neigh_offsets,
#     values_pp,
#     const_a_pp,
#     const_b_pp,
#     const_c_pp,
#     const_d_pp,
#     n_cells,
#     seed,
# ):
#     """
#     Stoichiometric probabilistic nucleation.
#     One successful event consumes threshold_inward oxidants from (i,j,k) and
#     threshold_outward active particles from local valid neighbours, then adds one product.
#     """
#     np.random.seed(seed)
#     n2 = n_cells * n_cells
#     n_active = active_check_offsets.shape[0]
#     thr_in = int(threshold_inward)
#     thr_out = int(threshold_outward)

#     for k in seed_slab_k:
#         for i in plane_indexes:
#             for j in range(n_cells):
#                 if full_3d[i, j, k]:
#                     continue
#                 c_max = int(oxidant[i, j, k] // thr_in)
#                 if c_max <= 0:
#                     continue
#                 idx_o = i + n_cells * j + n2 * k
#                 for _ in range(c_max):
#                     if product[i, j, k] >= ox_num:
#                         break
#                     if oxidant[i, j, k] < thr_in:
#                         break

#                     valid_count = 0
#                     total_active = 0
#                     _ni = np.empty(n_active, dtype=np.intp)
#                     _nj = np.empty(n_active, dtype=np.intp)
#                     _nk = np.empty(n_active, dtype=np.intp)
#                     _nidx = np.empty(n_active, dtype=np.intp)
#                     _ac = np.empty(n_active, dtype=np.int32)
#                     for row in active_check_offsets:
#                         di, dj, dk = int(row[0]), int(row[1]), int(row[2])
#                         ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
#                             i + di, j + dj, k + dk, n_cells
#                         )
#                         if valid and active[ii, jj, kk] > 0:
#                             cnt = int(active[ii, jj, kk])
#                             _ni[valid_count] = ii
#                             _nj[valid_count] = jj
#                             _nk[valid_count] = kk
#                             _nidx[valid_count] = ii + n_cells * jj + n2 * kk
#                             _ac[valid_count] = cnt
#                             total_active += cnt
#                             valid_count += 1

#                     if thr_out > 0 and total_active < thr_out:
#                         continue

#                     flat_count = 0
#                     for row in flat_neigh_offsets:
#                         di, dj, dk = int(row[0]), int(row[1]), int(row[2])
#                         ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
#                             i + di, j + dj, k + dk, n_cells
#                         )
#                         if valid and product_init[ii, jj, kk] > 0:
#                             flat_count += 1

#                     if flat_count == 0:
#                         prob = values_pp[k]
#                     else:
#                         prob = (
#                             const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k])
#                             + const_d_pp[k]
#                         )
#                     if np.random.random() >= prob:
#                         continue

#                     for _ in range(thr_out):
#                         picked_idx = 0
#                         while _ac[picked_idx] <= 0:
#                             picked_idx += 1

#                         ni = _ni[picked_idx]
#                         nj = _nj[picked_idx]
#                         nk = _nk[picked_idx]
#                         nidx = _nidx[picked_idx]
#                         slot_a = active[ni, nj, nk] - 1
#                         active_dirs[nidx, slot_a] = 0
#                         active[ni, nj, nk] -= 1
#                         _ac[picked_idx] -= 1

#                     old_o = int(oxidant[i, j, k])
#                     new_o = old_o - thr_in
#                     oxidant_dirs[idx_o, new_o:old_o] = 0
#                     oxidant[i, j, k] = new_o

#                     product[i, j, k] += 1
#                     if product[i, j, k] >= ox_num:
#                         full_3d[i, j, k] = True
#                     product_x_nzs[k] = True


# @numba.njit(fastmath=True, cache=_CACHE)
# def nucleation_subblock_kernel_simple_stoich(
#     oxidant,
#     oxidant_dirs,
#     active,
#     active_dirs,
#     product,
#     full_3d,
#     product_x_nzs,
#     ox_num,
#     threshold_inward,
#     threshold_outward,
#     seed_slab_k,
#     plane_indexes,
#     active_check_offsets,
#     n_cells,
#     seed,
# ):
#     """
#     Stoichiometric simplified nucleation (no probability term).
#     One successful event consumes threshold_inward oxidants from (i,j,k) and
#     threshold_outward active particles from local valid neighbours, then adds one product.
#     """
#     np.random.seed(seed)
#     n2 = n_cells * n_cells
#     n_active = active_check_offsets.shape[0]
#     thr_in = int(threshold_inward)
#     thr_out = int(threshold_outward)

#     for k in seed_slab_k:
#         for i in plane_indexes:
#             for j in range(n_cells):
#                 if full_3d[i, j, k]:
#                     continue
#                 c_max = int(oxidant[i, j, k] // thr_in)
#                 if c_max <= 0:
#                     continue
#                 idx_o = i + n_cells * j + n2 * k
#                 for _ in range(c_max):
#                     if product[i, j, k] >= ox_num:
#                         break
#                     if oxidant[i, j, k] < thr_in:
#                         break

#                     valid_count = 0
#                     total_active = 0
#                     _ni = np.empty(n_active, dtype=np.intp)
#                     _nj = np.empty(n_active, dtype=np.intp)
#                     _nk = np.empty(n_active, dtype=np.intp)
#                     _nidx = np.empty(n_active, dtype=np.intp)
#                     _ac = np.empty(n_active, dtype=np.int32)
#                     for row in active_check_offsets:
#                         di, dj, dk = int(row[0]), int(row[1]), int(row[2])
#                         ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
#                             i + di, j + dj, k + dk, n_cells
#                         )
#                         if valid and active[ii, jj, kk] > 0:
#                             cnt = int(active[ii, jj, kk])
#                             _ni[valid_count] = ii
#                             _nj[valid_count] = jj
#                             _nk[valid_count] = kk
#                             _nidx[valid_count] = ii + n_cells * jj + n2 * kk
#                             _ac[valid_count] = cnt
#                             total_active += cnt
#                             valid_count += 1

#                     if total_active < thr_out:
#                         continue
#                     for _ in range(thr_out):
#                         picked_idx = 0
#                         while _ac[picked_idx] <= 0:
#                             picked_idx += 1

#                         ni = _ni[picked_idx]
#                         nj = _nj[picked_idx]
#                         nk = _nk[picked_idx]
#                         nidx = _nidx[picked_idx]
#                         slot_a = active[ni, nj, nk] - 1
#                         active_dirs[nidx, slot_a] = 0
#                         active[ni, nj, nk] -= 1
#                         _ac[picked_idx] -= 1

#                     old_o = int(oxidant[i, j, k])
#                     new_o = old_o - thr_in
#                     oxidant_dirs[idx_o, new_o:old_o] = 0
#                     oxidant[i, j, k] = new_o

#                     product[i, j, k] += 1
#                     if product[i, j, k] >= ox_num:
#                         full_3d[i, j, k] = True
#                     product_x_nzs[k] = True


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_stoich_owner(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product_init,
    product_state,
    phase_id,
    ox_num,
    threshold_inward,
    threshold_outward,
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
    seed,
):
    np.random.seed(seed)
    n2 = n_cells * n_cells
    n_active = active_check_offsets.shape[0]
    owner_phase = product_state[0]
    state_count = product_state[1]
    thr_in = int(threshold_inward)
    thr_out = int(threshold_outward)
    pid = np.uint8(phase_id)

    for k in seed_slab_k:
        for i in plane_indexes:
            for j in range(n_cells):
                owner = owner_phase[i, j, k]
                if owner != 0 and owner != pid:
                    continue
                c_max = int(oxidant[i, j, k] // thr_in)
                if c_max <= 0:
                    continue
                idx_o = i + n_cells * j + n2 * k
                for _ in range(c_max):
                    cnt_here = int(state_count[i, j, k])
                    if cnt_here >= ox_num or oxidant[i, j, k] < thr_in:
                        break
                    if owner != 0 and owner != pid:
                        break

                    valid_count = 0
                    total_active = 0
                    _ni = np.empty(n_active, dtype=np.intp)
                    _nj = np.empty(n_active, dtype=np.intp)
                    _nk = np.empty(n_active, dtype=np.intp)
                    _nidx = np.empty(n_active, dtype=np.intp)
                    _ac = np.empty(n_active, dtype=np.int32)
                    for row in active_check_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
                            i + di, j + dj, k + dk, n_cells
                        )
                        if valid and active[ii, jj, kk] > 0:
                            cnt = int(active[ii, jj, kk])
                            _ni[valid_count] = ii
                            _nj[valid_count] = jj
                            _nk[valid_count] = kk
                            _nidx[valid_count] = ii + n_cells * jj + n2 * kk
                            _ac[valid_count] = cnt
                            total_active += cnt
                            valid_count += 1
                    if thr_out > 0 and total_active < thr_out:
                        continue

                    flat_count = 0
                    for row in flat_neigh_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
                            i + di, j + dj, k + dk, n_cells
                        )
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = (
                            const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k])
                            + const_d_pp[k]
                        )
                    if np.random.random() >= prob:
                        continue

                    for _ in range(thr_out):
                        picked_idx = 0
                        while _ac[picked_idx] <= 0:
                            picked_idx += 1
                        ni = _ni[picked_idx]
                        nj = _nj[picked_idx]
                        nk = _nk[picked_idx]
                        nidx = _nidx[picked_idx]
                        slot_a = active[ni, nj, nk] - 1
                        active_dirs[nidx, slot_a] = 0
                        active[ni, nj, nk] -= 1
                        _ac[picked_idx] -= 1

                    old_o = int(oxidant[i, j, k])
                    new_o = old_o - thr_in
                    oxidant_dirs[idx_o, new_o:old_o] = 0
                    oxidant[i, j, k] = new_o
                    if owner == 0:
                        owner_phase[i, j, k] = pid
                    state_count[i, j, k] = np.uint8(cnt_here + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_simple_stoich_owner(
    oxidant,
    oxidant_dirs,
    active,
    active_dirs,
    product_init,
    product_state,
    phase_id,
    ox_num,
    threshold_inward,
    threshold_outward,
    seed_slab_k,
    plane_indexes,
    active_check_offsets,
    n_cells,
    seed,
):
    np.random.seed(seed)
    n2 = n_cells * n_cells
    n_active = active_check_offsets.shape[0]
    owner_phase = product_state[0]
    state_count = product_state[1]
    thr_in = int(threshold_inward)
    thr_out = int(threshold_outward)
    pid = np.uint8(phase_id)

    for k in seed_slab_k:
        for i in plane_indexes:
            for j in range(n_cells):
                owner = owner_phase[i, j, k]
                if owner != 0 and owner != pid:
                    continue
                c_max = int(oxidant[i, j, k] // thr_in)
                if c_max <= 0:
                    continue
                idx_o = i + n_cells * j + n2 * k
                for _ in range(c_max):
                    cnt_here = int(state_count[i, j, k])
                    if cnt_here >= ox_num or oxidant[i, j, k] < thr_in:
                        break
                    if owner != 0 and owner != pid:
                        break

                    valid_count = 0
                    total_active = 0
                    _ni = np.empty(n_active, dtype=np.intp)
                    _nj = np.empty(n_active, dtype=np.intp)
                    _nk = np.empty(n_active, dtype=np.intp)
                    _nidx = np.empty(n_active, dtype=np.intp)
                    _ac = np.empty(n_active, dtype=np.int32)
                    for row in active_check_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
                            i + di, j + dj, k + dk, n_cells
                        )
                        if valid and active[ii, jj, kk] > 0:
                            cnt = int(active[ii, jj, kk])
                            _ni[valid_count] = ii
                            _nj[valid_count] = jj
                            _nk[valid_count] = kk
                            _nidx[valid_count] = ii + n_cells * jj + n2 * kk
                            _ac[valid_count] = cnt
                            total_active += cnt
                            valid_count += 1
                    if total_active < thr_out:
                        continue

                    for _ in range(thr_out):
                        picked_idx = 0
                        while _ac[picked_idx] <= 0:
                            picked_idx += 1
                        ni = _ni[picked_idx]
                        nj = _nj[picked_idx]
                        nk = _nk[picked_idx]
                        nidx = _nidx[picked_idx]
                        slot_a = active[ni, nj, nk] - 1
                        active_dirs[nidx, slot_a] = 0
                        active[ni, nj, nk] -= 1
                        _ac[picked_idx] -= 1

                    old_o = int(oxidant[i, j, k])
                    new_o = old_o - thr_in
                    oxidant_dirs[idx_o, new_o:old_o] = 0
                    oxidant[i, j, k] = new_o
                    if owner == 0:
                        owner_phase[i, j, k] = pid
                    state_count[i, j, k] = np.uint8(cnt_here + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def _dissolution_add_particles_at_cell(
    oxidant_count,
    oxidant_dirs,
    active_count,
    active_dirs,
    coords_list,
    n_cells,
    dissolution_thresholds,
    max_per_cell_ox,
    max_per_cell_active,
    packed_dirs,
):
    """Add dissolution_thresholds[0] oxidant and [1] active at each (i,j,k) in coords_list. Dirs from packed_dirs (6 face dirs)."""
    threshold_inward = int(dissolution_thresholds[0])
    threshold_outward = int(dissolution_thresholds[1])
    n2 = n_cells * n_cells
    for coord in coords_list:
        i, j, k = int(coord[0]), int(coord[1]), int(coord[2])
        nidx = i + n_cells * j + n2 * k
        slot_ox = int(oxidant_count[nidx])
        slot_act = int(active_count[nidx])
        cap_ox = max_per_cell_ox - slot_ox
        if cap_ox > 0:
            add_ox = threshold_inward if threshold_inward < cap_ox else cap_ox
            for _ in range(add_ox):
                r = np.random.randint(0, 6)
                oxidant_dirs[nidx, slot_ox] = packed_dirs[r]
                slot_ox += 1
        cap_act = max_per_cell_active - slot_act
        if cap_act > 0:
            add_act = threshold_outward if threshold_outward < cap_act else cap_act
            for _ in range(add_act):
                r = np.random.randint(0, 6)
                active_dirs[nidx, slot_act] = packed_dirs[r]
                slot_act += 1
        oxidant_count[nidx] = slot_ox
        active_count[nidx] = slot_act


@numba.njit(fastmath=True, cache=_CACHE)
def dissolution_subblock_kernel_snapshot(
    product_read,
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
    max_per_cell_ox,
    max_per_cell_active,
    packed_dirs,
):
    """
    Snapshot-based dissolution. Processes only k in [k_lo, k_hi] (z-slab). When a particle
    dissolves, appends (i,j,k) to a local buffer; at the end adds particles for all collected coords.
    """
    np.random.seed(seed)
    n_i, n_j, _ = product.shape
    coords_list = [(0, 0, 0) for _ in range(0)]  # empty list of (i,j,k) for Numba typing

    for k in range(k_lo, k_hi + 1):
        for idx_i in range(plane_indexes.shape[0]):
            i = int(plane_indexes[idx_i])
            for j in range(n_j):
                n_p = int(product[i, j, k])
                if n_p <= 0:
                    continue
                flat_count = 0
                for ni in range(6):
                    di = int(offsets_26[ni, 0])
                    dj = int(offsets_26[ni, 1])
                    dk = int(offsets_26[ni, 2])
                    ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
                        i + di, j + dj, k + dk, n_cells
                    )
                    if valid and product_read[ii, jj, kk] > 0:
                        flat_count += 1
                if flat_count == 0:
                    prob = values_pp[k]
                else:
                    prob = (
                        const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k])
                        + const_d_pp[k]
                    )
                for _ in range(n_p):
                    if np.random.random() < prob:
                        product[i, j, k] -= 1
                        coords_list.append((i, j, k))
                if product[i, j, k] <= 0 and k < full_3d.shape[2]:
                    full_3d[i, j, k] = False

    if len(coords_list) == 0:
        return
    _dissolution_add_particles_at_cell(
        oxidant_count,
        oxidant_dirs,
        active_count,
        active_dirs,
        coords_list,
        n_cells,
        dissolution_thresholds,
        max_per_cell_ox,
        max_per_cell_active,
        packed_dirs,
    )


# ---------------------------------------------------------------------------
# Dissolution subblock with snapshot + block detection (26-neighbour, prob /= bsf if in block)
# ---------------------------------------------------------------------------


@numba.njit(fastmath=True, cache=_CACHE)
def _dissolution_is_block_cell(neigh26_bool, block_patterns):
    """Return True if for any row in block_patterns, all 7 indexed entries in neigh26_bool are True."""
    for p in range(block_patterns.shape[0]):
        all7 = True
        for idx in range(7):
            if not neigh26_bool[block_patterns[p, idx]]:
                all7 = False
                break
        if all7:
            return True
    return False


@numba.njit(fastmath=True, cache=_CACHE)
def _dissolution_fill_neigh_26(neigh26, offsets_26, i, j, k, n_cells, product_read):
    """Fill all 26 entries of neigh26 from offsets_26 in one pass; return flat_count (sum of product_read for first 6)."""
    flat_count = 0
    for ni in range(26):
        di = int(offsets_26[ni, 0])
        dj = int(offsets_26[ni, 1])
        dk = int(offsets_26[ni, 2])
        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(
            i + di, j + dj, k + dk, n_cells
        )
        has_neigh = valid and product_read[ii, jj, kk] > 0
        neigh26[ni] = has_neigh
        if ni < 6 and has_neigh:
            flat_count += 1
    return flat_count


@numba.njit(fastmath=True, cache=_CACHE)
def dissolution_subblock_kernel_snapshot_with_blocks(
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
    block_patterns,
    bsf,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    n_z,
    seed,
    dissolution_thresholds,
    max_per_cell_ox,
    max_per_cell_active,
    packed_dirs,
):
    """
    Snapshot-based dissolution with block detection. Processes only k in [k_lo, k_hi] (z-slab).
    When a particle dissolves, appends (i,j,k) to a local buffer; at the end adds particles.
    """
    np.random.seed(seed)
    n_i, n_j, _ = product.shape
    bsf_inv = 1.0 / bsf if bsf > 0.0 else 1.0
    coords_list = [(0, 0, 0) for _ in range(0)]  # empty list of (i,j,k) for Numba typing

    for k in range(k_lo, k_hi + 1):
        for idx_i in range(plane_indexes.shape[0]):
            i = int(plane_indexes[idx_i])
            for j in range(n_j):
                n_p = int(product[i, j, k])
                if n_p <= 0:
                    continue
                neigh26 = np.zeros(26, dtype=np.bool_)
                flat_count = _dissolution_fill_neigh_26(
                    neigh26, offsets_26, i, j, k, n_cells, product_read
                )
                is_block = block_patterns.shape[0] > 0 and _dissolution_is_block_cell(neigh26, block_patterns)
                if flat_count == 0:
                    prob = values_pp[k]
                else:
                    prob = (
                        const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k])
                        + const_d_pp[k]
                    )
                if is_block:
                    prob *= bsf_inv
                for _ in range(n_p):
                    if np.random.random() < prob:
                        product[i, j, k] -= 1
                        coords_list.append((i, j, k))
                if product[i, j, k] <= 0 and k < full_3d.shape[2]:
                    full_3d[i, j, k] = False

    if len(coords_list) == 0:
        return
    _dissolution_add_particles_at_cell(
        oxidant_count,
        oxidant_dirs,
        active_count,
        active_dirs,
        coords_list,
        n_cells,
        dissolution_thresholds,
        max_per_cell_ox,
        max_per_cell_active,
        packed_dirs,
    )


@numba.njit(fastmath=True, cache=_CACHE)
def dissolution_subblock_kernel_snapshot_owner(
    product_state,
    phase_id,
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
    seed,
    dissolution_thresholds,
    max_per_cell_ox,
    max_per_cell_active,
    packed_dirs,
):
    np.random.seed(seed)
    n_i, n_j, _ = n_cells
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    coords_list = [(0, 0, 0) for _ in range(0)]

    for k in range(k_lo, k_hi + 1):
        for idx_i in range(plane_indexes.shape[0]):
            i = int(plane_indexes[idx_i])
            for j in range(n_j):
                if owner_phase[i, j, k] != pid:
                    continue
                n_p = int(state_count[i, j, k])
                if n_p <= 0:
                    continue
                flat_count = 0
                for ni in range(6):
                    di = int(offsets_26[ni, 0])
                    dj = int(offsets_26[ni, 1])
                    dk = int(offsets_26[ni, 2])
                    ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                    if valid and owner_phase[ii, jj, kk] > 0:
                        flat_count += 1
                if flat_count == 0:
                    prob = values_pp[k]
                else:
                    prob = (
                        const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k])
                        + const_d_pp[k]
                    )
                for _ in range(n_p):
                    if np.random.random() < prob:
                        state_count[i, j, k] -= 1
                        coords_list.append((i, j, k))
                if state_count[i, j, k] <= 0 and owner_phase[i, j, k] == pid:
                    owner_phase[i, j, k] = np.uint8(0)

    if len(coords_list) == 0:
        return
    _dissolution_add_particles_at_cell(
        oxidant_count,
        oxidant_dirs,
        active_count,
        active_dirs,
        coords_list,
        n_cells,
        dissolution_thresholds,
        max_per_cell_ox,
        max_per_cell_active,
        packed_dirs,
    )


@numba.njit(fastmath=True, cache=_CACHE)
def dissolution_subblock_kernel_snapshot_with_blocks_owner(
    product_state,
    phase_id,
    oxidant_count,
    oxidant_dirs,
    active_count,
    active_dirs,
    plane_indexes,
    offsets_26,
    k_lo,
    k_hi,
    block_patterns,
    bsf,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    dissolution_thresholds,
    max_per_cell_ox,
    max_per_cell_active,
    packed_dirs,
):
    np.random.seed(seed)
    n_i, n_j, _ = n_cells, n_cells, n_cells
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    bsf_inv = 1.0 / bsf if bsf > 0.0 else 1.0
    coords_list = [(0, 0, 0) for _ in range(0)]

    for k in range(k_lo, k_hi + 1):
        for idx_i in range(plane_indexes.shape[0]):
            i = int(plane_indexes[idx_i])
            for j in range(n_j):
                if owner_phase[i, j, k] != pid:
                    continue
                n_p = int(state_count[i, j, k])
                if n_p <= 0:
                    continue
                neigh26 = np.zeros(26, dtype=np.bool_)
                flat_count = 0
                for ni in range(26):
                    di = int(offsets_26[ni, 0])
                    dj = int(offsets_26[ni, 1])
                    dk = int(offsets_26[ni, 2])
                    ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                    has_neigh = valid and owner_phase[ii, jj, kk] > 0
                    neigh26[ni] = has_neigh
                    if ni < 6 and has_neigh:
                        flat_count += 1
                is_block = block_patterns.shape[0] > 0 and _dissolution_is_block_cell(neigh26, block_patterns)
                if flat_count == 0:
                    prob = values_pp[k]
                else:
                    prob = (
                        const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k])
                        + const_d_pp[k]
                    )
                if is_block:
                    prob *= bsf_inv
                for _ in range(n_p):
                    if np.random.random() < prob:
                        state_count[i, j, k] -= 1
                        coords_list.append((i, j, k))
                if state_count[i, j, k] <= 0 and owner_phase[i, j, k] == pid:
                    owner_phase[i, j, k] = np.uint8(0)

    if len(coords_list) == 0:
        return
    _dissolution_add_particles_at_cell(
        oxidant_count,
        oxidant_dirs,
        active_count,
        active_dirs,
        coords_list,
        n_cells,
        dissolution_thresholds,
        max_per_cell_ox,
        max_per_cell_active,
        packed_dirs,
    )
