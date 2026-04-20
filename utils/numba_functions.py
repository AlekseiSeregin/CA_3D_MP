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
    n = owner_phase.shape[0]
    for i in range(ub + 1):
        s = 0
        for j in range(n):
            for k in range(n):
                if owner_phase[i, j, k] == pid:
                    s += int(state_count[i, j, k])
        out[i] = s
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
def product_counts_blocks_ignited_from_state(
    owner_phase,
    state_count,
    pid_to_row,
    n_products,
    Bx,
    By,
    Bz,
    n_blocks,
    block_cells_x,
    block_cells_y,
    block_cells_z,
    x_hi,
):
    """
    Per-block product counts for *multiple* phase ids in one scan, limited to ignited x-range.

    - owner_phase, state_count: unified product_state views (shape (n,n,n), dtype uint8)
    - pid_to_row: int16 array of length 256 mapping phase_id -> row index in [0, n_products), else -1
    - n_products: number of tracked product phases (rows in output)
    - Bx,By,Bz: number of blocks along each axis
    - n_blocks: total blocks Bx*By*Bz (second dimension of output); must match Bx,By,Bz
    - block_cells_x/y/z: cells per block along each axis
    - x_hi: exclusive upper bound in x to scan; should be multiple of block_cells_x

    Returns: uint32 array shape (n_products, n_blocks), flattened block id order:
      block_id = (bx * By + by) * Bz + bz
    Only blocks with bx < x_hi//block_cells_x receive non-zero counts.
    """
    n = owner_phase.shape[0]
    By_tot = int(By)
    Bz_tot = int(Bz)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)
    xh = int(x_hi)
    n_blk = int(n_blocks)
    out = np.zeros((int(n_products), n_blk), dtype=np.uint32)
    for i in range(xh):
        bx = i // cx
        for j in range(n):
            by = j // cy
            for k in range(n):
                bz = k // cz
                pid = int(owner_phase[i, j, k])
                row = int(pid_to_row[pid])
                if row < 0:
                    continue
                blk = (bx * By_tot + by) * Bz_tot + bz
                out[row, blk] += np.uint32(state_count[i, j, k])
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
def severe_planes_clear_product_release(
    owner_phase,
    state_count,
    phase_id,
    plane_indexes,
    n_i,
    n_j,
    n_z,
    oxidant_count,
    oxidant_dirs,
    active_count,
    active_dirs,
    dissolution_thresholds,
    max_per_cell_ox,
    max_per_cell_active,
    packed_dirs,
    seed,
):
    """
    For each x-index in plane_indexes: clear all cells on that x-slice where owner_phase == pid
    (state_count zeroed, owner zeroed). For each removed state_count unit at (i,j,k), add
    dissolution_thresholds[0] inward and [1] outward particles at the same cell (flat layout
    i + n_i*j + n_i*n_j*k), matching add_dissolution_particles_to_grid per event.
    """
    threshold_inward = int(dissolution_thresholds[0])
    threshold_outward = int(dissolution_thresholds[1])
    np.random.seed(seed)
    n2 = n_i * n_j
    n_packed = packed_dirs.shape[0]
    pid = np.uint8(phase_id)

    for p in range(plane_indexes.shape[0]):
        plane_i = int(plane_indexes[p])
        for j in range(n_j):
            for k in range(n_z):
                if owner_phase[plane_i, j, k] != pid:
                    continue
                c = int(state_count[plane_i, j, k])
                owner_phase[plane_i, j, k] = np.uint8(0)
                state_count[plane_i, j, k] = np.uint8(0)
                if c <= 0:
                    continue
                nidx = plane_i + n_i * j + n2 * k
                for _ in range(c):
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
def severe_blocks_clear_product_release(
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
    max_per_cell_ox,
    max_per_cell_active,
    packed_dirs,
    seed,
    By,
    Bz,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """
    Clear product within each block_id subvolume and release inward/outward particles per unit state_count.
    block_id order: (bx*bpa + by)*bpa + bz.
    """
    np.random.seed(seed)
    n2 = n_i * n_j
    threshold_inward = int(dissolution_thresholds[0])
    threshold_outward = int(dissolution_thresholds[1])
    pid8 = np.uint8(pid)
    By_tot = int(By)
    Bz_tot = int(Bz)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)
    n_packed = packed_dirs.shape[0]
    for bi in range(block_ids.shape[0]):
        bid = int(block_ids[bi])
        bx = bid // (By_tot * Bz_tot)
        rem = bid - bx * (By_tot * Bz_tot)
        by = rem // Bz_tot
        bz = rem - by * Bz_tot
        i_lo = bx * cx
        i_hi = i_lo + cx
        j_lo = by * cy
        j_hi = j_lo + cy
        k_lo = bz * cz
        k_hi = k_lo + cz
        for i in range(i_lo, i_hi):
            for j in range(j_lo, j_hi):
                for k in range(k_lo, k_hi):
                    if owner_phase[i, j, k] != pid8:
                        continue
                    c = int(state_count[i, j, k])
                    owner_phase[i, j, k] = np.uint8(0)
                    state_count[i, j, k] = np.uint8(0)
                    if c <= 0:
                        continue
                    nidx = i + n_i * j + n2 * k
                    for _ in range(c):
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
                # (blockmask variants gate here)
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
def nucleation_subblock_kernel_owner_blockmask(
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
    seed,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Owner-aware legacy probabilistic nucleation gated by (bx,by,bz) bitmask."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    n_active = active_check_offsets.shape[0]
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in seed_slab_k:
        bz = k // cz
        for i in plane_indexes:
            bx = i // cx
            for j in range(n_cells):
                by = j // cy
                if (int(block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
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
                    valid_count = 0
                    _ni = np.empty(n_active, dtype=np.intp)
                    _nj = np.empty(n_active, dtype=np.intp)
                    _nk = np.empty(n_active, dtype=np.intp)
                    for row in active_check_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
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
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k]) + const_d_pp[k]
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


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_simple_owner_blockmask(
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
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Owner-aware legacy simplified nucleation gated by (bx,by,bz) bitmask."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    n_active = active_check_offsets.shape[0]
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in seed_slab_k:
        bz = k // cz
        for i in plane_indexes:
            bx = i // cx
            for j in range(n_cells):
                by = j // cy
                if (int(block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
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
                    valid_count = 0
                    _ni = np.empty(n_active, dtype=np.intp)
                    _nj = np.empty(n_active, dtype=np.intp)
                    _nk = np.empty(n_active, dtype=np.intp)
                    for row in active_check_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
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
def nucleation_subblock_kernel_stoich_owner_blockmask(
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
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Threshold probabilistic nucleation gated by (bx,by,bz) bitmask."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    n_active = active_check_offsets.shape[0]
    owner_phase = product_state[0]
    state_count = product_state[1]
    thr_in = int(threshold_inward)
    thr_out = int(threshold_outward)
    pid = np.uint8(phase_id)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in seed_slab_k:
        bz = k // cz
        for i in plane_indexes:
            bx = i // cx
            for j in range(n_cells):
                by = j // cy
                if (int(block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
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
                    valid_count = 0
                    total_active = 0
                    _ni = np.empty(n_active, dtype=np.intp)
                    _nj = np.empty(n_active, dtype=np.intp)
                    _nk = np.empty(n_active, dtype=np.intp)
                    _nidx = np.empty(n_active, dtype=np.intp)
                    _ac = np.empty(n_active, dtype=np.int32)
                    for row in active_check_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
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
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k]) + const_d_pp[k]
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
def nucleation_subblock_kernel_simple_stoich_owner_blockmask(
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
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Threshold simplified nucleation gated by (bx,by,bz) bitmask."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    n_active = active_check_offsets.shape[0]
    owner_phase = product_state[0]
    state_count = product_state[1]
    thr_in = int(threshold_inward)
    thr_out = int(threshold_outward)
    pid = np.uint8(phase_id)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in seed_slab_k:
        bz = k // cz
        for i in plane_indexes:
            bx = i // cx
            for j in range(n_cells):
                by = j // cy
                if (int(block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
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

                    valid_count = 0
                    total_active = 0
                    _ni = np.empty(n_active, dtype=np.intp)
                    _nj = np.empty(n_active, dtype=np.intp)
                    _nk = np.empty(n_active, dtype=np.intp)
                    _nidx = np.empty(n_active, dtype=np.intp)
                    _ac = np.empty(n_active, dtype=np.int32)
                    for row in active_check_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
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
def nucleation_subblock_kernel_owner_spec(
    oxidant,
    oxidant_dirs,
    product_init,
    product_state,
    phase_id,
    ox_num,
    seed_slab_k,
    plane_indexes,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
):
    """Owner-aware legacy probabilistic nucleation for no-outward products (1 inward : 0 outward)."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
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
                    cnt_here = int(state_count[i, j, k])
                    if cnt_here >= ox_num or oxidant[i, j, k] <= 0:
                        break

                    flat_count = 0
                    for row in flat_neigh_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k]) + const_d_pp[k]
                    if np.random.random() >= prob:
                        continue

                    slot_o = oxidant[i, j, k] - 1
                    oxidant_dirs[idx_o, slot_o] = 0
                    oxidant[i, j, k] -= 1
                    if owner == 0:
                        owner_phase[i, j, k] = pid
                    state_count[i, j, k] = np.uint8(cnt_here + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_owner_spec_blockmask(
    oxidant,
    oxidant_dirs,
    product_init,
    product_state,
    phase_id,
    ox_num,
    seed_slab_k,
    plane_indexes,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Owner-aware legacy probabilistic nucleation for no-outward products gated by block bitmask."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in seed_slab_k:
        bz = k // cz
        for i in plane_indexes:
            bx = i // cx
            for j in range(n_cells):
                by = j // cy
                if (int(block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
                owner = owner_phase[i, j, k]
                if owner != 0 and owner != pid:
                    continue
                c_max = int(oxidant[i, j, k])
                if c_max <= 0:
                    continue
                idx_o = i + n_cells * j + n2 * k
                for _ in range(c_max):
                    cnt_here = int(state_count[i, j, k])
                    if cnt_here >= ox_num or oxidant[i, j, k] <= 0:
                        break

                    flat_count = 0
                    for row in flat_neigh_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k]) + const_d_pp[k]
                    if np.random.random() >= prob:
                        continue

                    slot_o = oxidant[i, j, k] - 1
                    oxidant_dirs[idx_o, slot_o] = 0
                    oxidant[i, j, k] -= 1
                    if owner == 0:
                        owner_phase[i, j, k] = pid
                    state_count[i, j, k] = np.uint8(cnt_here + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_simple_owner_spec(
    oxidant,
    oxidant_dirs,
    product_state,
    phase_id,
    ox_num,
    seed_slab_k,
    plane_indexes,
    n_cells,
    seed,
):
    """Owner-aware legacy simplified nucleation for no-outward products (1 inward : 0 outward)."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
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
                    cnt_here = int(state_count[i, j, k])
                    if cnt_here >= ox_num or oxidant[i, j, k] <= 0:
                        break

                    slot_o = oxidant[i, j, k] - 1
                    oxidant_dirs[idx_o, slot_o] = 0
                    oxidant[i, j, k] -= 1
                    if owner == 0:
                        owner_phase[i, j, k] = pid
                    state_count[i, j, k] = np.uint8(cnt_here + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_simple_owner_spec_blockmask(
    oxidant,
    oxidant_dirs,
    product_state,
    phase_id,
    ox_num,
    seed_slab_k,
    plane_indexes,
    n_cells,
    seed,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Owner-aware legacy simplified nucleation for no-outward products gated by block bitmask."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in seed_slab_k:
        bz = k // cz
        for i in plane_indexes:
            bx = i // cx
            for j in range(n_cells):
                by = j // cy
                if (int(block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
                owner = owner_phase[i, j, k]
                if owner != 0 and owner != pid:
                    continue
                c_max = int(oxidant[i, j, k])
                if c_max <= 0:
                    continue
                idx_o = i + n_cells * j + n2 * k
                for _ in range(c_max):
                    cnt_here = int(state_count[i, j, k])
                    if cnt_here >= ox_num or oxidant[i, j, k] <= 0:
                        break

                    slot_o = oxidant[i, j, k] - 1
                    oxidant_dirs[idx_o, slot_o] = 0
                    oxidant[i, j, k] -= 1
                    if owner == 0:
                        owner_phase[i, j, k] = pid
                    state_count[i, j, k] = np.uint8(cnt_here + 1)



@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_stoich_owner_spec(
    oxidant,
    oxidant_dirs,
    product_init,
    product_state,
    phase_id,
    ox_num,
    threshold_inward,
    seed_slab_k,
    plane_indexes,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
):
    """Owner-aware threshold probabilistic nucleation for no-outward products (thr_in : 0 outward)."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    thr_in = int(threshold_inward)
 
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

                    flat_count = 0
                    for row in flat_neigh_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k]) + const_d_pp[k]
                    if np.random.random() >= prob:
                        continue

                    old_o = int(oxidant[i, j, k])
                    new_o = old_o - thr_in
                    oxidant_dirs[idx_o, new_o:old_o] = 0
                    oxidant[i, j, k] = new_o
                    if owner == 0:
                        owner_phase[i, j, k] = pid
                    state_count[i, j, k] = np.uint8(cnt_here + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_stoich_owner_spec_blockmask(
    oxidant,
    oxidant_dirs,
    product_init,
    product_state,
    phase_id,
    ox_num,
    threshold_inward,
    seed_slab_k,
    plane_indexes,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Owner-aware threshold probabilistic nucleation for no-outward products gated by block bitmask."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    thr_in = int(threshold_inward)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in seed_slab_k:
        bz = k // cz
        for i in plane_indexes:
            bx = i // cx
            for j in range(n_cells):
                by = j // cy
                if (int(block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
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

                    flat_count = 0
                    for row in flat_neigh_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k]) + const_d_pp[k]
                    if np.random.random() >= prob:
                        continue

                    old_o = int(oxidant[i, j, k])
                    new_o = old_o - thr_in
                    oxidant_dirs[idx_o, new_o:old_o] = 0
                    oxidant[i, j, k] = new_o
                    if owner == 0:
                        owner_phase[i, j, k] = pid
                    state_count[i, j, k] = np.uint8(cnt_here + 1)



@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_simple_stoich_owner_spec(
    oxidant,
    oxidant_dirs,
    product_state,
    phase_id,
    ox_num,
    threshold_inward,
    seed_slab_k,
    plane_indexes,
    n_cells,
    seed,
):
    """Owner-aware threshold simplified nucleation for no-outward products (thr_in : 0 outward)."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    thr_in = int(threshold_inward)

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

                    old_o = int(oxidant[i, j, k])
                    new_o = old_o - thr_in
                    oxidant_dirs[idx_o, new_o:old_o] = 0
                    oxidant[i, j, k] = new_o
                    if owner == 0:
                        owner_phase[i, j, k] = pid
                    state_count[i, j, k] = np.uint8(cnt_here + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_simple_stoich_owner_spec_blockmask(
    oxidant,
    oxidant_dirs,
    product_state,
    phase_id,
    ox_num,
    threshold_inward,
    seed_slab_k,
    plane_indexes,
    n_cells,
    seed,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Owner-aware threshold simplified nucleation for no-outward products gated by block bitmask."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    thr_in = int(threshold_inward)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in seed_slab_k:
        bz = k // cz
        for i in plane_indexes:
            bx = i // cx
            for j in range(n_cells):
                by = j // cy
                if (int(block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
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

                    old_o = int(oxidant[i, j, k])
                    new_o = old_o - thr_in
                    oxidant_dirs[idx_o, new_o:old_o] = 0
                    oxidant[i, j, k] = new_o
                    if owner == 0:
                        owner_phase[i, j, k] = pid
                    state_count[i, j, k] = np.uint8(cnt_here + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def _nucleation_fold_pick_target(i, j, k, n_cells, ox_num, pid, owner_phase, state_count):
    """
    Among center (i,j,k) and 6 face neighbors (PBC in y,z; hard x), pick the cell with the largest
    state_count for this product (owner 0 or pid) that is not yet full (count < ox_num).
    Tie-break: first scanned wins (center, +x, -x, +y, -y, +z, -z).
    Returns (ti, tj, tk) or (-1, -1, -1) if no valid site.
    """
    best_i = -1
    best_j = -1
    best_k = -1
    best_c = -1
    for t in range(7):
        if t == 0:
            di, dj, dk = 0, 0, 0
        elif t == 1:
            di, dj, dk = 1, 0, 0
        elif t == 2:
            di, dj, dk = -1, 0, 0
        elif t == 3:
            di, dj, dk = 0, 1, 0
        elif t == 4:
            di, dj, dk = 0, -1, 0
        elif t == 5:
            di, dj, dk = 0, 0, 1
        else:
            di, dj, dk = 0, 0, -1
        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
        if not valid:
            continue
        ow = owner_phase[ii, jj, kk]
        if ow != 0 and ow != pid:
            continue
        c = int(state_count[ii, jj, kk])
        if c >= ox_num:
            continue
        if c > best_c:
            best_c = c
            best_i = ii
            best_j = jj
            best_k = kk
    return best_i, best_j, best_k


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_owner_fold(
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
    seed,
):
    """Probabilistic owner nucleation with fold placement when product neighbours exist (snapshot)."""
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
                    if oxidant[i, j, k] <= 0:
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

                    if flat_count == 0:
                        ti, tj, tk = i, j, k
                    else:
                        ti, tj, tk = _nucleation_fold_pick_target(
                            i, j, k, n_cells, ox_num, pid, owner_phase, state_count
                        )
                        if ti < 0:
                            ti, tj, tk = i, j, k
                    ow_t = owner_phase[ti, tj, tk]
                    if ow_t != 0 and ow_t != pid:
                        continue
                    if int(state_count[ti, tj, tk]) >= ox_num:
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

                    if owner_phase[ti, tj, tk] == 0:
                        owner_phase[ti, tj, tk] = pid
                    state_count[ti, tj, tk] = np.uint8(int(state_count[ti, tj, tk]) + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_owner_fold_blockmask(
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
    seed,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Fold owner nucleation gated by (bx,by,bz) bitmask."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    n_active = active_check_offsets.shape[0]
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in seed_slab_k:
        bz = k // cz
        for i in plane_indexes:
            bx = i // cx
            for j in range(n_cells):
                by = j // cy
                if (int(block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
                owner = owner_phase[i, j, k]
                if owner != 0 and owner != pid:
                    continue
                c_max = int(oxidant[i, j, k])
                if c_max <= 0:
                    continue
                idx_o = i + n_cells * j + n2 * k
                for _ in range(c_max):
                    if oxidant[i, j, k] <= 0:
                        break
                    valid_count = 0
                    _ni = np.empty(n_active, dtype=np.intp)
                    _nj = np.empty(n_active, dtype=np.intp)
                    _nk = np.empty(n_active, dtype=np.intp)
                    for row in active_check_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
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
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k]) + const_d_pp[k]
                    if np.random.random() >= prob:
                        continue

                    if flat_count == 0:
                        ti, tj, tk = i, j, k
                    else:
                        ti, tj, tk = _nucleation_fold_pick_target(i, j, k, n_cells, ox_num, pid, owner_phase, state_count)
                        if ti < 0:
                            ti, tj, tk = i, j, k
                    ow_t = owner_phase[ti, tj, tk]
                    if ow_t != 0 and ow_t != pid:
                        continue
                    if int(state_count[ti, tj, tk]) >= ox_num:
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

                    if owner_phase[ti, tj, tk] == 0:
                        owner_phase[ti, tj, tk] = pid
                    state_count[ti, tj, tk] = np.uint8(int(state_count[ti, tj, tk]) + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_stoich_owner_fold(
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
                    if oxidant[i, j, k] < thr_in:
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

                    if flat_count == 0:
                        ti, tj, tk = i, j, k
                    else:
                        ti, tj, tk = _nucleation_fold_pick_target(
                            i, j, k, n_cells, ox_num, pid, owner_phase, state_count
                        )
                        if ti < 0:
                            ti, tj, tk = i, j, k
                    ow_t = owner_phase[ti, tj, tk]
                    if ow_t != 0 and ow_t != pid:
                        continue
                    if int(state_count[ti, tj, tk]) >= ox_num:
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
                    cnt_t = int(state_count[ti, tj, tk])
                    if owner_phase[ti, tj, tk] == 0:
                        owner_phase[ti, tj, tk] = pid
                    state_count[ti, tj, tk] = np.uint8(cnt_t + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_stoich_owner_fold_blockmask(
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
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Fold stoich owner nucleation gated by (bx,by,bz) bitmask."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    n_active = active_check_offsets.shape[0]
    owner_phase = product_state[0]
    state_count = product_state[1]
    thr_in = int(threshold_inward)
    thr_out = int(threshold_outward)
    pid = np.uint8(phase_id)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in seed_slab_k:
        bz = k // cz
        for i in plane_indexes:
            bx = i // cx
            for j in range(n_cells):
                by = j // cy
                if (int(block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
                owner = owner_phase[i, j, k]
                if owner != 0 and owner != pid:
                    continue
                c_max = int(oxidant[i, j, k] // thr_in)
                if c_max <= 0:
                    continue
                idx_o = i + n_cells * j + n2 * k
                for _ in range(c_max):
                    if oxidant[i, j, k] < thr_in:
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
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
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
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k]) + const_d_pp[k]
                    if np.random.random() >= prob:
                        continue

                    if flat_count == 0:
                        ti, tj, tk = i, j, k
                    else:
                        ti, tj, tk = _nucleation_fold_pick_target(i, j, k, n_cells, ox_num, pid, owner_phase, state_count)
                        if ti < 0:
                            ti, tj, tk = i, j, k
                    ow_t = owner_phase[ti, tj, tk]
                    if ow_t != 0 and ow_t != pid:
                        continue
                    if int(state_count[ti, tj, tk]) >= ox_num:
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
                    cnt_t = int(state_count[ti, tj, tk])
                    if owner_phase[ti, tj, tk] == 0:
                        owner_phase[ti, tj, tk] = pid
                    state_count[ti, tj, tk] = np.uint8(cnt_t + 1)



@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_owner_spec_fold(
    oxidant,
    oxidant_dirs,
    product_init,
    product_state,
    phase_id,
    ox_num,
    seed_slab_k,
    plane_indexes,
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
                    if oxidant[i, j, k] <= 0:
                        break
                    flat_count = 0
                    for row in flat_neigh_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k]) + const_d_pp[k]
                    if np.random.random() >= prob:
                        continue

                    if flat_count == 0:
                        ti, tj, tk = i, j, k
                    else:
                        ti, tj, tk = _nucleation_fold_pick_target(
                            i, j, k, n_cells, ox_num, pid, owner_phase, state_count
                        )
                        if ti < 0:
                            ti, tj, tk = i, j, k
                    ow_t = owner_phase[ti, tj, tk]
                    if ow_t != 0 and ow_t != pid:
                        continue
                    if int(state_count[ti, tj, tk]) >= ox_num:
                        continue

                    slot_o = oxidant[i, j, k] - 1
                    oxidant_dirs[idx_o, slot_o] = 0
                    oxidant[i, j, k] -= 1
                    if owner_phase[ti, tj, tk] == 0:
                        owner_phase[ti, tj, tk] = pid
                    state_count[ti, tj, tk] = np.uint8(int(state_count[ti, tj, tk]) + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_owner_spec_fold_blockmask(
    oxidant,
    oxidant_dirs,
    product_init,
    product_state,
    phase_id,
    ox_num,
    seed_slab_k,
    plane_indexes,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Fold owner nucleation (no outward) gated by block bitmask."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in seed_slab_k:
        bz = k // cz
        for i in plane_indexes:
            bx = i // cx
            for j in range(n_cells):
                by = j // cy
                if (int(block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
                owner = owner_phase[i, j, k]
                if owner != 0 and owner != pid:
                    continue
                c_max = int(oxidant[i, j, k])
                if c_max <= 0:
                    continue
                idx_o = i + n_cells * j + n2 * k
                for _ in range(c_max):
                    if oxidant[i, j, k] <= 0:
                        break
                    flat_count = 0
                    for row in flat_neigh_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k]) + const_d_pp[k]
                    if np.random.random() >= prob:
                        continue

                    if flat_count == 0:
                        ti, tj, tk = i, j, k
                    else:
                        ti, tj, tk = _nucleation_fold_pick_target(i, j, k, n_cells, ox_num, pid, owner_phase, state_count)
                        if ti < 0:
                            ti, tj, tk = i, j, k
                    ow_t = owner_phase[ti, tj, tk]
                    if ow_t != 0 and ow_t != pid:
                        continue
                    if int(state_count[ti, tj, tk]) >= ox_num:
                        continue

                    slot_o = oxidant[i, j, k] - 1
                    oxidant_dirs[idx_o, slot_o] = 0
                    oxidant[i, j, k] -= 1
                    if owner_phase[ti, tj, tk] == 0:
                        owner_phase[ti, tj, tk] = pid
                    state_count[ti, tj, tk] = np.uint8(int(state_count[ti, tj, tk]) + 1)



@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_stoich_owner_spec_fold(
    oxidant,
    oxidant_dirs,
    product_init,
    product_state,
    phase_id,
    ox_num,
    threshold_inward,
    seed_slab_k,
    plane_indexes,
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
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    thr_in = int(threshold_inward)

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
                    if oxidant[i, j, k] < thr_in:
                        break
                    flat_count = 0
                    for row in flat_neigh_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k]) + const_d_pp[k]
                    if np.random.random() >= prob:
                        continue

                    if flat_count == 0:
                        ti, tj, tk = i, j, k
                    else:
                        ti, tj, tk = _nucleation_fold_pick_target(
                            i, j, k, n_cells, ox_num, pid, owner_phase, state_count
                        )
                        if ti < 0:
                            ti, tj, tk = i, j, k
                    ow_t = owner_phase[ti, tj, tk]
                    if ow_t != 0 and ow_t != pid:
                        continue
                    if int(state_count[ti, tj, tk]) >= ox_num:
                        continue

                    old_o = int(oxidant[i, j, k])
                    new_o = old_o - thr_in
                    oxidant_dirs[idx_o, new_o:old_o] = 0
                    oxidant[i, j, k] = new_o
                    cnt_t = int(state_count[ti, tj, tk])
                    if owner_phase[ti, tj, tk] == 0:
                        owner_phase[ti, tj, tk] = pid
                    state_count[ti, tj, tk] = np.uint8(cnt_t + 1)


@numba.njit(fastmath=True, cache=_CACHE)
def nucleation_subblock_kernel_stoich_owner_spec_fold_blockmask(
    oxidant,
    oxidant_dirs,
    product_init,
    product_state,
    phase_id,
    ox_num,
    threshold_inward,
    seed_slab_k,
    plane_indexes,
    flat_neigh_offsets,
    values_pp,
    const_a_pp,
    const_b_pp,
    const_c_pp,
    const_d_pp,
    n_cells,
    seed,
    block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Fold stoich nucleation (no outward) gated by block bitmask."""
    np.random.seed(seed)
    n2 = n_cells * n_cells
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    thr_in = int(threshold_inward)
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in seed_slab_k:
        bz = k // cz
        for i in plane_indexes:
            bx = i // cx
            for j in range(n_cells):
                by = j // cy
                if (int(block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
                owner = owner_phase[i, j, k]
                if owner != 0 and owner != pid:
                    continue
                c_max = int(oxidant[i, j, k] // thr_in)
                if c_max <= 0:
                    continue
                idx_o = i + n_cells * j + n2 * k
                for _ in range(c_max):
                    if oxidant[i, j, k] < thr_in:
                        break
                    flat_count = 0
                    for row in flat_neigh_offsets:
                        di, dj, dk = int(row[0]), int(row[1]), int(row[2])
                        ii, jj, kk, valid = _nucleation_subblock_apply_pbc(i + di, j + dj, k + dk, n_cells)
                        if valid and product_init[ii, jj, kk] > 0:
                            flat_count += 1
                    if flat_count == 0:
                        prob = values_pp[k]
                    else:
                        prob = const_a_pp[k] * np.exp(const_b_pp[k] * flat_count + const_c_pp[k]) + const_d_pp[k]
                    if np.random.random() >= prob:
                        continue

                    if flat_count == 0:
                        ti, tj, tk = i, j, k
                    else:
                        ti, tj, tk = _nucleation_fold_pick_target(i, j, k, n_cells, ox_num, pid, owner_phase, state_count)
                        if ti < 0:
                            ti, tj, tk = i, j, k
                    ow_t = owner_phase[ti, tj, tk]
                    if ow_t != 0 and ow_t != pid:
                        continue
                    if int(state_count[ti, tj, tk]) >= ox_num:
                        continue

                    old_o = int(oxidant[i, j, k])
                    new_o = old_o - thr_in
                    oxidant_dirs[idx_o, new_o:old_o] = 0
                    oxidant[i, j, k] = new_o
                    cnt_t = int(state_count[ti, tj, tk])
                    if owner_phase[ti, tj, tk] == 0:
                        owner_phase[ti, tj, tk] = pid
                    state_count[ti, tj, tk] = np.uint8(cnt_t + 1)


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
    n_i = n_cells
    n_j = n_cells
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
                    # Dissolution neighbourhood is phase-local: only same product owner counts.
                    if valid and owner_phase[ii, jj, kk] == pid:
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
                    # Dissolution neighbourhood is phase-local: only same product owner counts.
                    has_neigh = valid and owner_phase[ii, jj, kk] == pid
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


@numba.njit(fastmath=True, cache=_CACHE)
def dissolution_subblock_kernel_snapshot_owner_blockmask(
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
    dissolution_block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Like dissolution_subblock_kernel_snapshot_owner but only at (bx,by,bz) with a dissolution mask bit."""
    np.random.seed(seed)
    n_i = n_cells
    n_j = n_cells
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    coords_list = [(0, 0, 0) for _ in range(0)]
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in range(k_lo, k_hi + 1):
        bz = k // cz
        for idx_i in range(plane_indexes.shape[0]):
            i = int(plane_indexes[idx_i])
            bx = i // cx
            for j in range(n_j):
                by = j // cy
                if (int(dissolution_block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
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
                    if valid and owner_phase[ii, jj, kk] == pid:
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
def dissolution_subblock_kernel_snapshot_with_blocks_owner_blockmask(
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
    dissolution_block_mask_bits,
    block_cells_x,
    block_cells_y,
    block_cells_z,
):
    """Like dissolution_subblock_kernel_snapshot_with_blocks_owner with per-block dissolution mask."""
    np.random.seed(seed)
    n_i, n_j, _ = n_cells, n_cells, n_cells
    owner_phase = product_state[0]
    state_count = product_state[1]
    pid = np.uint8(phase_id)
    bsf_inv = 1.0 / bsf if bsf > 0.0 else 1.0
    coords_list = [(0, 0, 0) for _ in range(0)]
    cx = int(block_cells_x)
    cy = int(block_cells_y)
    cz = int(block_cells_z)

    for k in range(k_lo, k_hi + 1):
        bz = k // cz
        for idx_i in range(plane_indexes.shape[0]):
            i = int(plane_indexes[idx_i])
            bx = i // cx
            for j in range(n_j):
                by = j // cy
                if (int(dissolution_block_mask_bits[bx, by]) >> bz) & 1 == 0:
                    continue
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
                    has_neigh = valid and owner_phase[ii, jj, kk] == pid
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
