import time
import threading
import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import numba
from configuration import Config


# Boundary condition constants (per axis)
BC_PERIODIC = 0
BC_REFLECTION = 1
BC_DELETION = 2

_DIRS_6 = np.array([
    [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]
], dtype=np.int8)


def _pack_dir(dx, dy, dz):
    """Pack (dx, dy, dz) each in {-1, 0, 1} into one byte (2 bits per component)."""
    return (int(dx) + 1) + ((int(dy) + 1) << 2) + ((int(dz) + 1) << 4)


# Precomputed packed form of _DIRS_6 for initial placement
_DIRS_6_PACKED = np.array([_pack_dir(d[0], d[1], d[2]) for d in _DIRS_6], dtype=np.uint8)

# Flat index for 3D (i, j, k) in C order: i changes fastest
# Use Python ints to avoid overflow with numpy int32 when n is large (e.g. n² or n³ > 2^31)
def _idx(i, j, k, n):
    ni, nj, nk, nn = int(i), int(j), int(k), int(n)
    return ni + nn * (nj + nn * nk)

# ---------------------------------------------------------------------------
# Subblock partition: split along z-axis with 2-cell gaps (like x-gaps) to avoid
# write overlap: particles move ±1 in z so neighbours must not write into same k.
# x iteration is limited to 0..x_max per step so inward/sparse-x does not waste workers.
# ---------------------------------------------------------------------------

def _partition_domain_z(n, n_blocks):
    """
    Partition z-axis (k) into n_blocks interior ranges with 2-cell gaps between blocks.
    Interior blocks only cover k in [1, n-2] so that with periodic BC in z:
    - no particle at k=n-1 (boundary) is in an interior block, so no worker writes to k=0 (wrap);
    - no particle at k=0 is in an interior block, so no worker writes to k=n-1 (wrap).
    k=0 and k=n-1 are always in gap_z_set and are processed by gap workers only.
    Returns (z_ranges, gap_z_set): list of (k_lo, k_hi) and set of gap k indices.
    """
    if n_blocks <= 0 or n < 3:
        return [], set(range(n))
    gap_width = 2
    # Interior indices 1..n-2 only (exclude 0 and n-1 for periodic z safety)
    interior_size = n - 2
    total_gaps = (n_blocks - 1) * gap_width
    interior_total = interior_size - total_gaps
    if interior_total < n_blocks:
        return [], set(range(n))
    base = interior_total // n_blocks
    extra = interior_total % n_blocks
    z_ranges = []
    k = 1
    for b in range(n_blocks):
        size = base + (1 if b < extra else 0)
        if size <= 0 or k > n - 2:
            break
        k_hi = min(k + size - 1, n - 2)
        z_ranges.append((k, k_hi))
        k = k_hi + 1 + gap_width
    gap_z_set = set(ki for ki in range(n) if not any(a <= ki <= b for a, b in z_ranges))
    return z_ranges, gap_z_set


# ---------------------------------------------------------------------------
# x_max from count: max x (i) where any particle exists (for inward / sparse-x optimization)
# ---------------------------------------------------------------------------

def _compute_x_max_from_count(read_count, n):
    """
    From flat count array (idx = i + n*j + n2*k), return max i such that count[idx] > 0 for some j,k.
    Returns -1 if no particles. Used to avoid iterating over empty x in diffusion kernels.
    """
    nonzero = np.flatnonzero(read_count > 0)
    if nonzero.size == 0:
        return -1
    return int(np.max(nonzero % n))


# ---------------------------------------------------------------------------
# Shared buffer layout: one segment = [count (n³ int8)][dirs (n³·max_per_cell uint8 packed)]
# Packed dirs: one byte per (dx,dy,dz) with dx,dy,dz in {-1,0,1}: byte = (dx+1)|((dy+1)<<2)|((dz+1)<<4)
# ---------------------------------------------------------------------------

def _views_from_segment(shm, n, max_per_cell, count_bytes, dirs_bytes):
    """Return (count, dirs) as numpy views on the shared segment (zero-copy). dirs are packed: (n3, max_per_cell) uint8."""
    n3 = n * n * n
    count_dtype = np.int8 if count_bytes == n3 else np.int32
    count = np.ndarray((n3,), dtype=count_dtype, buffer=shm.buf, offset=0)
    dirs = np.ndarray((n3, max_per_cell), dtype=np.uint8, buffer=shm.buf, offset=count_bytes)
    return count, dirs


# ---------------------------------------------------------------------------
# Numba-compiled kernels: three separate functions (x periodic / reflection / deletion)
# ---------------------------------------------------------------------------
# Boundary condition is fixed for the whole run. Three separate kernels with x-BC
# inlined (no branch). The right one is chosen from config (boundary_x) before the run.

# x periodic: no branch, just wrap. Loop over z-block (k_lo..k_hi), then j, then i in 0..x_max.
@numba.njit(fastmath=True, cache=True)
def _diffuse_subblock_kernel_x_periodic(
    read_count, read_dirs, write_count, write_dirs,
    k_lo, k_hi, x_max, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(k_lo, k_hi + 1):
        for j in range(n):
            for i in range(0, x_max + 1):
                idx = i + n * j + n2 * k
                nc = read_count[idx]
                if nc == 0:
                    continue
                nc = min(nc, max_per_cell)
                for c in range(nc):
                    b = read_dirs[idx, c]
                    d0 = int(b & 3) - 1
                    d1 = int((b >> 2) & 3) - 1
                    d2 = int((b >> 4) & 3) - 1
                    r = np.random.random()
                    if r <= p1:
                        nd0, nd1, nd2 = d2, d0, d1
                    elif r <= p2:
                        nd0, nd1, nd2 = -d2, -d0, -d1
                    elif r <= p3:
                        nd0, nd1, nd2 = d1, d2, d0
                    elif r <= p4:
                        nd0, nd1, nd2 = -d1, -d2, -d0
                    elif r <= p_r:
                        nd0, nd1, nd2 = -d0, -d1, -d2
                    else:
                        nd0, nd1, nd2 = d0, d1, d2
                    nx = ((i + nd0) % n + n) % n
                    ndx = nd0
                    ny = ((j + nd1) % n + n) % n
                    nz = ((k + nd2) % n + n) % n
                    ndy, ndz = nd1, nd2
                    nidx = nx + n * ny + n2 * nz
                    slot = write_count[nidx]
                    if slot >= max_per_cell:
                        continue
                    write_dirs[nidx, slot] = (ndx + 1) + (ndy + 1) * 4 + (ndz + 1) * 16
                    write_count[nidx] = slot + 1

# x reflection: clamp and flip
@numba.njit(fastmath=True, cache=True)
def _diffuse_subblock_kernel_x_reflection(
    read_count, read_dirs, write_count, write_dirs,
    k_lo, k_hi, x_max, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(k_lo, k_hi + 1):
        for j in range(n):
            for i in range(0, x_max + 1):
                idx = i + n * j + n2 * k
                nc = read_count[idx]
                if nc == 0:
                    continue
                # Safety: clamp nc to max_per_cell to prevent reading beyond array bounds
                nc = min(nc, max_per_cell)
                for c in range(nc):
                    b = read_dirs[idx, c]
                    d0 = int(b & 3) - 1
                    d1 = int((b >> 2) & 3) - 1
                    d2 = int((b >> 4) & 3) - 1
                    r = np.random.random()
                    if r <= p1:
                        nd0, nd1, nd2 = d2, d0, d1
                    elif r <= p2:
                        nd0, nd1, nd2 = -d2, -d0, -d1
                    elif r <= p3:
                        nd0, nd1, nd2 = d1, d2, d0
                    elif r <= p4:
                        nd0, nd1, nd2 = -d1, -d2, -d0
                    elif r <= p_r:
                        nd0, nd1, nd2 = -d0, -d1, -d2
                    else:
                        nd0, nd1, nd2 = d0, d1, d2
                    v_new = i + nd0
                    if v_new < 0:
                        nx, ndx = 0, -nd0
                    elif v_new >= n:
                        nx, ndx = n - 1, -nd0
                    else:
                        nx, ndx = v_new, nd0
                    ny = ((j + nd1) % n + n) % n
                    nz = ((k + nd2) % n + n) % n
                    ndy, ndz = nd1, nd2
                    nidx = nx + n * ny + n2 * nz
                    slot = write_count[nidx]
                    if slot >= max_per_cell:
                        continue
                    write_dirs[nidx, slot] = (ndx + 1) + (ndy + 1) * 4 + (ndz + 1) * 16
                    write_count[nidx] = slot + 1

# x deletion: clamp, skip write if out of bounds
@numba.njit(fastmath=True, cache=True)
def _diffuse_subblock_kernel_x_deletion(
    read_count, read_dirs, write_count, write_dirs,
    k_lo, k_hi, x_max, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    # Partition by z (k_lo..k_hi); iterate i in 0..x_max only (early out for sparse x).
    np.random.seed(seed)
    n2 = n * n
    for k in range(k_lo, k_hi + 1):
        for j in range(n):
            for i in range(0, x_max + 1):
                idx = i + n * j + n2 * k
                nc = read_count[idx]
                if nc == 0:
                    continue
                # Safety: clamp nc to max_per_cell to prevent reading beyond array bounds
                nc = min(nc, max_per_cell)
                for c in range(nc):
                    b = read_dirs[idx, c]
                    d0 = int(b & 3) - 1
                    d1 = int((b >> 2) & 3) - 1
                    d2 = int((b >> 4) & 3) - 1
                    r = np.random.random()
                    if r <= p1:
                        nd0, nd1, nd2 = d2, d0, d1
                    elif r <= p2:
                        nd0, nd1, nd2 = -d2, -d0, -d1
                    elif r <= p3:
                        nd0, nd1, nd2 = d1, d2, d0
                    elif r <= p4:
                        nd0, nd1, nd2 = -d1, -d2, -d0
                    elif r <= p_r:
                        nd0, nd1, nd2 = -d0, -d1, -d2
                    else:
                        nd0, nd1, nd2 = d0, d1, d2
                    v_new = i + nd0
                    if v_new < 0 or v_new >= n:
                        continue
                    nx, ndx = v_new, nd0
                    ny = ((j + nd1) % n + n) % n
                    nz = ((k + nd2) % n + n) % n
                    ndy, ndz = nd1, nd2
                    nidx = nx + n * ny + n2 * nz
                    slot = write_count[nidx]
                    if slot >= max_per_cell:
                        continue
                    write_dirs[nidx, slot] = (ndx + 1) + (ndy + 1) * 4 + (ndz + 1) * 16
                    write_count[nidx] = slot + 1
    

# Per-side x BC: one kernel per (left, right) combination; no BC branches in hot path.
# p=periodic(0), r=reflection(1), d=deletion(2). First suffix=left, second=right.

# (0,1) left periodic, right reflection
@numba.njit(fastmath=True, cache=True)
def _diffuse_subblock_kernel_x_pr(
    read_count, read_dirs, write_count, write_dirs,
    k_lo, k_hi, x_max, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(k_lo, k_hi + 1):
        for j in range(n):
            for i in range(0, x_max + 1):
                idx = i + n * j + n2 * k
                nc = read_count[idx]
                if nc == 0:
                    continue
                nc = min(nc, max_per_cell)
                for c in range(nc):
                    b = read_dirs[idx, c]
                    d0 = int(b & 3) - 1
                    d1 = int((b >> 2) & 3) - 1
                    d2 = int((b >> 4) & 3) - 1
                    r = np.random.random()
                    if r <= p1:
                        nd0, nd1, nd2 = d2, d0, d1
                    elif r <= p2:
                        nd0, nd1, nd2 = -d2, -d0, -d1
                    elif r <= p3:
                        nd0, nd1, nd2 = d1, d2, d0
                    elif r <= p4:
                        nd0, nd1, nd2 = -d1, -d2, -d0
                    elif r <= p_r:
                        nd0, nd1, nd2 = -d0, -d1, -d2
                    else:
                        nd0, nd1, nd2 = d0, d1, d2
                    v_new = i + nd0
                    if v_new < 0:
                        nx, ndx = n - 1, nd0
                    elif v_new >= n:
                        nx, ndx = n - 1, -nd0
                    else:
                        nx, ndx = v_new, nd0
                    ny = ((j + nd1) % n + n) % n
                    nz = ((k + nd2) % n + n) % n
                    ndy, ndz = nd1, nd2
                    nidx = nx + n * ny + n2 * nz
                    slot = write_count[nidx]
                    if slot >= max_per_cell:
                        continue
                    write_dirs[nidx, slot] = (ndx + 1) + (ndy + 1) * 4 + (ndz + 1) * 16
                    write_count[nidx] = slot + 1

# (0,2) left periodic, right deletion
@numba.njit(fastmath=True, cache=True)
def _diffuse_subblock_kernel_x_pd(
    read_count, read_dirs, write_count, write_dirs,
    k_lo, k_hi, x_max, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(k_lo, k_hi + 1):
        for j in range(n):
            for i in range(0, x_max + 1):
                idx = i + n * j + n2 * k
                nc = read_count[idx]
                if nc == 0:
                    continue
                nc = min(nc, max_per_cell)
                for c in range(nc):
                    b = read_dirs[idx, c]
                    d0 = int(b & 3) - 1
                    d1 = int((b >> 2) & 3) - 1
                    d2 = int((b >> 4) & 3) - 1
                    r = np.random.random()
                    if r <= p1:
                        nd0, nd1, nd2 = d2, d0, d1
                    elif r <= p2:
                        nd0, nd1, nd2 = -d2, -d0, -d1
                    elif r <= p3:
                        nd0, nd1, nd2 = d1, d2, d0
                    elif r <= p4:
                        nd0, nd1, nd2 = -d1, -d2, -d0
                    elif r <= p_r:
                        nd0, nd1, nd2 = -d0, -d1, -d2
                    else:
                        nd0, nd1, nd2 = d0, d1, d2
                    v_new = i + nd0
                    if v_new >= n:
                        continue
                    if v_new < 0:
                        nx, ndx = n - 1, nd0
                    else:
                        nx, ndx = v_new, nd0
                    ny = ((j + nd1) % n + n) % n
                    nz = ((k + nd2) % n + n) % n
                    ndy, ndz = nd1, nd2
                    nidx = nx + n * ny + n2 * nz
                    slot = write_count[nidx]
                    if slot >= max_per_cell:
                        continue
                    write_dirs[nidx, slot] = (ndx + 1) + (ndy + 1) * 4 + (ndz + 1) * 16
                    write_count[nidx] = slot + 1

# (1,0) left reflection, right periodic
@numba.njit(fastmath=True, cache=True)
def _diffuse_subblock_kernel_x_rp(
    read_count, read_dirs, write_count, write_dirs,
    k_lo, k_hi, x_max, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(k_lo, k_hi + 1):
        for j in range(n):
            for i in range(0, x_max + 1):
                idx = i + n * j + n2 * k
                nc = read_count[idx]
                if nc == 0:
                    continue
                nc = min(nc, max_per_cell)
                for c in range(nc):
                    b = read_dirs[idx, c]
                    d0 = int(b & 3) - 1
                    d1 = int((b >> 2) & 3) - 1
                    d2 = int((b >> 4) & 3) - 1
                    r = np.random.random()
                    if r <= p1:
                        nd0, nd1, nd2 = d2, d0, d1
                    elif r <= p2:
                        nd0, nd1, nd2 = -d2, -d0, -d1
                    elif r <= p3:
                        nd0, nd1, nd2 = d1, d2, d0
                    elif r <= p4:
                        nd0, nd1, nd2 = -d1, -d2, -d0
                    elif r <= p_r:
                        nd0, nd1, nd2 = -d0, -d1, -d2
                    else:
                        nd0, nd1, nd2 = d0, d1, d2
                    v_new = i + nd0
                    if v_new < 0:
                        nx, ndx = 0, -nd0
                    elif v_new >= n:
                        nx, ndx = 0, nd0
                    else:
                        nx, ndx = v_new, nd0
                    ny = ((j + nd1) % n + n) % n
                    nz = ((k + nd2) % n + n) % n
                    ndy, ndz = nd1, nd2
                    nidx = nx + n * ny + n2 * nz
                    slot = write_count[nidx]
                    if slot >= max_per_cell:
                        continue
                    write_dirs[nidx, slot] = (ndx + 1) + (ndy + 1) * 4 + (ndz + 1) * 16
                    write_count[nidx] = slot + 1

# (1,2) left reflection, right deletion
@numba.njit(fastmath=True, cache=True)
def _diffuse_subblock_kernel_x_rd(
    read_count, read_dirs, write_count, write_dirs,
    k_lo, k_hi, x_max, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(k_lo, k_hi + 1):
        for j in range(n):
            for i in range(0, x_max + 1):
                idx = i + n * j + n2 * k
                nc = read_count[idx]
                if nc == 0:
                    continue
                nc = min(nc, max_per_cell)
                for c in range(nc):
                    b = read_dirs[idx, c]
                    d0 = int(b & 3) - 1
                    d1 = int((b >> 2) & 3) - 1
                    d2 = int((b >> 4) & 3) - 1
                    r = np.random.random()
                    if r <= p1:
                        nd0, nd1, nd2 = d2, d0, d1
                    elif r <= p2:
                        nd0, nd1, nd2 = -d2, -d0, -d1
                    elif r <= p3:
                        nd0, nd1, nd2 = d1, d2, d0
                    elif r <= p4:
                        nd0, nd1, nd2 = -d1, -d2, -d0
                    elif r <= p_r:
                        nd0, nd1, nd2 = -d0, -d1, -d2
                    else:
                        nd0, nd1, nd2 = d0, d1, d2
                    v_new = i + nd0
                    if v_new >= n:
                        continue
                    if v_new < 0:
                        nx, ndx = 0, -nd0
                    else:
                        nx, ndx = v_new, nd0
                    ny = ((j + nd1) % n + n) % n
                    nz = ((k + nd2) % n + n) % n
                    ndy, ndz = nd1, nd2
                    nidx = nx + n * ny + n2 * nz
                    slot = write_count[nidx]
                    if slot >= max_per_cell:
                        continue
                    write_dirs[nidx, slot] = (ndx + 1) + (ndy + 1) * 4 + (ndz + 1) * 16
                    write_count[nidx] = slot + 1

# (2,0) left deletion, right periodic
@numba.njit(fastmath=True, cache=True)
def _diffuse_subblock_kernel_x_dp(
    read_count, read_dirs, write_count, write_dirs,
    k_lo, k_hi, x_max, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(k_lo, k_hi + 1):
        for j in range(n):
            for i in range(0, x_max + 1):
                idx = i + n * j + n2 * k
                nc = read_count[idx]
                if nc == 0:
                    continue
                nc = min(nc, max_per_cell)
                for c in range(nc):
                    b = read_dirs[idx, c]
                    d0 = int(b & 3) - 1
                    d1 = int((b >> 2) & 3) - 1
                    d2 = int((b >> 4) & 3) - 1
                    r = np.random.random()
                    if r <= p1:
                        nd0, nd1, nd2 = d2, d0, d1
                    elif r <= p2:
                        nd0, nd1, nd2 = -d2, -d0, -d1
                    elif r <= p3:
                        nd0, nd1, nd2 = d1, d2, d0
                    elif r <= p4:
                        nd0, nd1, nd2 = -d1, -d2, -d0
                    elif r <= p_r:
                        nd0, nd1, nd2 = -d0, -d1, -d2
                    else:
                        nd0, nd1, nd2 = d0, d1, d2
                    v_new = i + nd0
                    if v_new < 0:
                        continue
                    if v_new >= n:
                        nx, ndx = 0, nd0
                    else:
                        nx, ndx = v_new, nd0
                    ny = ((j + nd1) % n + n) % n
                    nz = ((k + nd2) % n + n) % n
                    ndy, ndz = nd1, nd2
                    nidx = nx + n * ny + n2 * nz
                    slot = write_count[nidx]
                    if slot >= max_per_cell:
                        continue
                    write_dirs[nidx, slot] = (ndx + 1) + (ndy + 1) * 4 + (ndz + 1) * 16
                    write_count[nidx] = slot + 1

# (2,1) left deletion, right reflection
@numba.njit(fastmath=True, cache=True)
def _diffuse_subblock_kernel_x_dr(
    read_count, read_dirs, write_count, write_dirs,
    k_lo, k_hi, x_max, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(k_lo, k_hi + 1):
        for j in range(n):
            for i in range(0, x_max + 1):
                idx = i + n * j + n2 * k
                nc = read_count[idx]
                if nc == 0:
                    continue
                nc = min(nc, max_per_cell)
                for c in range(nc):
                    b = read_dirs[idx, c]
                    d0 = int(b & 3) - 1
                    d1 = int((b >> 2) & 3) - 1
                    d2 = int((b >> 4) & 3) - 1
                    r = np.random.random()
                    if r <= p1:
                        nd0, nd1, nd2 = d2, d0, d1
                    elif r <= p2:
                        nd0, nd1, nd2 = -d2, -d0, -d1
                    elif r <= p3:
                        nd0, nd1, nd2 = d1, d2, d0
                    elif r <= p4:
                        nd0, nd1, nd2 = -d1, -d2, -d0
                    elif r <= p_r:
                        nd0, nd1, nd2 = -d0, -d1, -d2
                    else:
                        nd0, nd1, nd2 = d0, d1, d2
                    v_new = i + nd0
                    if v_new < 0:
                        continue
                    if v_new >= n:
                        nx, ndx = n - 1, -nd0
                    else:
                        nx, ndx = v_new, nd0
                    ny = ((j + nd1) % n + n) % n
                    nz = ((k + nd2) % n + n) % n
                    ndy, ndz = nd1, nd2
                    nidx = nx + n * ny + n2 * nz
                    slot = write_count[nidx]
                    if slot >= max_per_cell:
                        continue
                    write_dirs[nidx, slot] = (ndx + 1) + (ndy + 1) * 4 + (ndz + 1) * 16
                    write_count[nidx] = slot + 1

# Lookup: _BC_X_KERNELS_2D[bc_left][bc_right], bc in {0=periodic, 1=reflection, 2=deletion}
_BC_X_KERNELS_2D = (
    (_diffuse_subblock_kernel_x_periodic, _diffuse_subblock_kernel_x_pr, _diffuse_subblock_kernel_x_pd),
    (_diffuse_subblock_kernel_x_rp, _diffuse_subblock_kernel_x_reflection, _diffuse_subblock_kernel_x_rd),
    (_diffuse_subblock_kernel_x_dp, _diffuse_subblock_kernel_x_dr, _diffuse_subblock_kernel_x_deletion),
)


def _worker_subblock(args):
    """
    Worker: attach to read/write shared segments by name; use views only (zero-copy).
    Domain split by z (k_lo..k_hi); kernel iterates i in 0..x_max only (all workers have work).
    bc_left, bc_right: 0=periodic, 1=reflection, 2=deletion; select one of 9 kernels.
    """
    (read_name, write_name, n, max_per_cell, count_bytes, dirs_bytes,
     k_lo, k_hi, x_max, bc_left, bc_right, p1, p2, p3, p4, p_r, seed) = args

    shm_r = shared_memory.SharedMemory(name=read_name)
    shm_w = shared_memory.SharedMemory(name=write_name)
    read_count, read_dirs = _views_from_segment(shm_r, n, max_per_cell, count_bytes, dirs_bytes)
    write_count, write_dirs = _views_from_segment(shm_w, n, max_per_cell, count_bytes, dirs_bytes)

    kernel_func = _BC_X_KERNELS_2D[bc_left][bc_right]
    kernel_func(
        read_count, read_dirs, write_count, write_dirs,
        k_lo, k_hi, x_max, n, max_per_cell,
        p1, p2, p3, p4, p_r, seed
    )

    shm_r.close()
    shm_w.close()


def _partition_gap_z_parallel(gap_z_set, min_spacing=3, n_z=None):
    """
    Partition gap z-coordinates into groups that can be processed in parallel.
    Gap k's at least min_spacing apart have non-overlapping writable z-ranges
    (particles move by at most ±1 in z, so gap at k writes to k in {k-1, k, k+1}).
    If n_z is given, z is treated as periodic: distance between k=0 and k=n_z-1 is 1,
    so they are never placed in the same group (they would write to each other).
    Returns list of groups, each group is a list of k-coordinates safe to process in parallel.
    """
    if len(gap_z_set) == 0:
        return []
    gap_z_list = sorted(gap_z_set)

    def z_dist(a, b):
        d = abs(a - b)
        if n_z is not None and n_z > 0:
            d = min(d, n_z - d)
        return d

    groups = []
    used = set()
    for k in gap_z_list:
        if k in used:
            continue
        group = [k]
        used.add(k)
        for other in gap_z_list:
            if other in used:
                continue
            can_add = True
            for gk in group:
                if z_dist(other, gk) < min_spacing:
                    can_add = False
                    break
            if can_add:
                group.append(other)
                used.add(other)
        groups.append(group)
    return groups


def _worker_gap_z(args):
    """
    Worker for gap z: process one gap k (all i, j for that k). Uses same kernel with k_lo=k_hi=k_gap.
    """
    (read_name, write_name, n, max_per_cell, count_bytes, dirs_bytes,
     gap_k, x_max, bc_left, bc_right, p1, p2, p3, p4, p_r, seed) = args

    shm_r = shared_memory.SharedMemory(name=read_name)
    shm_w = shared_memory.SharedMemory(name=write_name)
    read_count, read_dirs = _views_from_segment(shm_r, n, max_per_cell, count_bytes, dirs_bytes)
    write_count, write_dirs = _views_from_segment(shm_w, n, max_per_cell, count_bytes, dirs_bytes)

    kernel_func = _BC_X_KERNELS_2D[bc_left][bc_right]
    kernel_func(
        read_count, read_dirs, write_count, write_dirs,
        gap_k, gap_k, x_max, n, max_per_cell,
        p1, p2, p3, p4, p_r, seed
    )

    shm_r.close()
    shm_w.close()


def _update_gap_z_parallel(read_name, write_name, n, max_per_cell, count_bytes, dirs_bytes,
                           gap_z_groups, bc_left, bc_right, p1, p2, p3, p4, p_r, pool, rng, x_max=-1):
    """
    Process gap z cells in parallel. If x_max >= 0 it is passed to the kernel (limit i to 0..x_max).
    """
    if len(gap_z_groups) == 0:
        return
    base_seed = rng.integers(0, 2**31)
    for group_idx, group in enumerate(gap_z_groups):
        args_list = [
            (read_name, write_name, n, max_per_cell, count_bytes, dirs_bytes,
             gap_k, x_max, bc_left, bc_right, p1, p2, p3, p4, p_r, base_seed + group_idx * 1000 + ki)
            for ki, gap_k in enumerate(group)
        ]
        pool.map(_worker_gap_z, args_list)


def diffuse_3d_one_step_shm(
    read_name, write_name, n, max_per_cell, count_bytes, dirs_bytes,
    subblock_arg_templates, gap_z_groups, bc_left, bc_right, p1, p2, p3, p4, p_r, pool, rng
):
    """
    One step: zero write buffer, compute x_max, run interior workers (read → write), then gap-z phase.
    Domain partitioned by z with 2-cell gaps; gap z-coordinates processed in parallel after interior.
    bc_left, bc_right: 0=periodic, 1=reflection, 2=deletion (per x-side).
    """
    # Zero write buffer
    shm_w = shared_memory.SharedMemory(name=write_name)
    write_count, _ = _views_from_segment(shm_w, n, max_per_cell, count_bytes, dirs_bytes)
    write_count.fill(0)
    shm_w.close()

    # Max x coordinate that has any particles (from flat count: i = idx % n)
    shm_r = shared_memory.SharedMemory(name=read_name)
    read_count, _ = _views_from_segment(shm_r, n, max_per_cell, count_bytes, dirs_bytes)
    x_max = _compute_x_max_from_count(read_count, n)
    shm_r.close()

    # Interior z-blocks (no write overlap: gaps between blocks)
    args_list = [
        (read_name, write_name, tpl[0], tpl[1], tpl[2], tpl[3], tpl[4], tpl[5], x_max, tpl[6], tpl[7], tpl[8], tpl[9], tpl[10], tpl[11], tpl[12], rng.integers(0, 2**31))
        for tpl in subblock_arg_templates
    ]
    pool.map(_worker_subblock, args_list)

    # Gap z cells (processed in groups so writable k-ranges don't overlap)
    _update_gap_z_parallel(
        read_name, write_name, n, max_per_cell, count_bytes, dirs_bytes,
        gap_z_groups, bc_left, bc_right, p1, p2, p3, p4, p_r, pool, rng, x_max
    )


def _parse_boundary(s):
    """Convert string to BC constant: 'periodic'|'reflection'|'deletion' (or 'open')."""
    v = str(s).strip().lower()
    if v in ("periodic", "p"):
        return BC_PERIODIC
    if v in ("reflection", "reflect", "r"):
        return BC_REFLECTION
    if v in ("deletion", "delete", "open", "d", "o"):
        return BC_DELETION
    raise ValueError(f"Unknown boundary condition: {s!r}. Use periodic, reflection, or deletion.")


# ---------------------------------------------------------------------------
# Shared Diffusion Parameters: Computed once, reused by all elements
# ---------------------------------------------------------------------------

def _boundary_from_config(element_type, side):
    """Get boundary string from Config for element_type ('outward'|'inward') and side ('left'|'right')."""
    if element_type == 'outward':
        key = f'DIFFUSION_BOUNDARY_X_OUTWARD_{side.upper()}'
    else:
        key = f'DIFFUSION_BOUNDARY_X_INWARD_{side.upper()}'
    return getattr(Config, key)


class DiffusionParameters:
    """
    Global shared parameters for diffusion (grid-independent, computed once).
    Boundary x is per-side: boundary_x_left, boundary_x_right (from element config or Config).
    """
    _instances = {}  # Cache by (n, max_per_cell, element_type, boundary_x_left, boundary_x_right, n_workers)
    
    def __init__(self, max_per_cell, element_type='outward', boundary_x_left=None, boundary_x_right=None):
        """
        Initialize shared diffusion parameters.
        
        Args:
            max_per_cell: maximum particles per cell (element-specific)
            element_type: 'outward' or 'inward'
            boundary_x_left: 'periodic'|'reflection'|'deletion' for x<0; if None, from Config
            boundary_x_right: same for x>=n; if None, from Config
        """
        
        self.n = Config.N_CELLS_PER_AXIS
        self.boundary_x_left = boundary_x_left if boundary_x_left is not None else _boundary_from_config(element_type, 'left')
        self.boundary_x_right = boundary_x_right if boundary_x_right is not None else _boundary_from_config(element_type, 'right')
        
        if element_type == 'outward':
            self.n_workers = getattr(Config, 'OUTWARD_DIFFUSION_WORKERS', 7)
        elif element_type == 'inward':
            self.n_workers = getattr(Config, 'INWARD_DIFFUSION_WORKERS', 3)
        else:
            raise ValueError(f"Unknown element_type: {element_type}. Use 'outward' or 'inward'.")
        
        self.max_per_cell = max_per_cell
        
        prep = prepare_diffusion_run(
            self.n, self.n_workers, max_per_cell,
            self.boundary_x_left, self.boundary_x_right, 0.0, 0.0
        )
        self.count_bytes = prep["count_bytes"]
        self.dirs_bytes = prep["dirs_bytes"]
        self.subblock_arg_templates_base = prep["subblock_arg_templates"]
        self.gap_z_groups = prep["gap_z_groups"]
        self.bc_left = prep["bc_left"]
        self.bc_right = prep["bc_right"]
    
    @classmethod
    def get_or_create(cls, max_per_cell, element_type='outward', boundary_x_left=None, boundary_x_right=None):
        """
        Get existing instance or create new one (singleton per configuration).
        boundary_x_left/right: if None, read from Config (DIFFUSION_BOUNDARY_X_*_LEFT/RIGHT) by element_type.
        """
        
        n = Config.N_CELLS_PER_AXIS
        if element_type == 'outward':
            n_workers = getattr(Config, 'OUTWARD_DIFFUSION_WORKERS', 7)
        elif element_type == 'inward':
            n_workers = getattr(Config, 'INWARD_DIFFUSION_WORKERS', 3)
        else:
            raise ValueError(f"Unknown element_type: {element_type}. Use 'outward' or 'inward'.")
        
        bl = boundary_x_left if boundary_x_left is not None else _boundary_from_config(element_type, 'left')
        br = boundary_x_right if boundary_x_right is not None else _boundary_from_config(element_type, 'right')
        key = (n, max_per_cell, element_type, bl, br, n_workers)
        if key not in cls._instances:
            cls._instances[key] = cls(max_per_cell, element_type, bl, br)
        return cls._instances[key]
    
    def get_subblock_templates(self, p1, p2, p3, p4, p_r):
        """Get subblock argument templates (z-ranges k_lo, k_hi) with element-specific p and bc."""
        return [
            (self.n, self.max_per_cell, self.count_bytes, self.dirs_bytes,
             tpl[4], tpl[5], self.bc_left, self.bc_right, p1, p2, p3, p4, p_r)
            for tpl in self.subblock_arg_templates_base
        ]


# ---------------------------------------------------------------------------
# DiffusionEngine: Applies diffusion to elements (like shuffling a Rubik's cube)
# ---------------------------------------------------------------------------

class DiffusionEngine:
    """
    Unified diffusion engine that applies Chopard-Droz diffusion to elements.
    
    Uses one dedicated process pool per element (not per type). So with 2 outward
    elements and 4 workers per outward, and 2 inward elements and 2 workers per inward,
    diffuse_multiple() runs 4+4+2+2 = 12 workers in parallel (no sequential step).
    
    Usage:
        n_out = Config.OUTWARD_DIFFUSION_WORKERS   # e.g. 4
        n_in  = Config.INWARD_DIFFUSION_WORKERS    # e.g. 2
        engine = DiffusionEngine(n_out, n_in, rng)
        engine.diffuse(element)                           # One element (creates temp pool)
        engine.diffuse_multiple([o1, o2, i1, i2])         # All in parallel: 4+4+2+2 workers
        engine.close()                                    # When done (closes cached pools)
    """
    
    def __init__(self, n_outward_workers, n_inward_workers, rng, worker_pools=None):
        """
        Initialize diffusion engine with worker counts per element and RNG.
        
        Args:
            n_outward_workers: number of processes per outward-diffusing element
            n_inward_workers: number of processes per inward-diffusing element
            rng: numpy random number generator (used once to seed thread-local RNGs when diffusing multiple elements in parallel)
        """
        self.n_outward_workers = n_outward_workers
        self.n_inward_workers = n_inward_workers
        self.rng = rng
        self.worker_pools = worker_pools
        self._pools_managed_externally = worker_pools is not None
        # One RNG per thread so when we run multiple elements in parallel (ThreadPoolExecutor) each has its own stream; no lock.
        self._rng_base_seed = int(rng.integers(0, 2**31))
        self._thread_local = threading.local()
        self._pools = None  # list of pools, one per element
        # Config (max_per_cell, element_type) is constant per element; boundaries come from Config in diffusion module.
        self._config_cache = {}  # id(element) -> config dict, filled once per element
        self._state_cache = {}   # id(element) -> state dict (read_name, write_name, p1..p_r); names updated after each swap
    
    def _get_cached_state(self, element):
        """Return diffusion state for element; filled once, then read/write names updated in cache after each swap."""
        eid = id(element)
        if eid not in self._state_cache:
            self._state_cache[eid] = dict(element.get_diffusion_state())
        return self._state_cache[eid]
    
    def _swap_cached_state_names(self, element):
        """Update cached state after element.swap_diffusion_buffers(): swap read_name and write_name in cache."""
        eid = id(element)
        if eid in self._state_cache:
            s = self._state_cache[eid]
            s['read_name'], s['write_name'] = s['write_name'], s['read_name']
    
    def _get_thread_rng(self):
        """Per-thread RNG so when multiple elements are diffused in parallel, each thread has its own; no lock."""
        if not hasattr(self._thread_local, 'rng'):
            self._thread_local.rng = np.random.default_rng(
                seed=(self._rng_base_seed + threading.get_ident()) % (2**32)
            )
        return self._thread_local.rng
    
    def _diffuse_with_pool(self, element, pool):
        """Apply one diffusion step to an element using the given dedicated pool."""
        if getattr(element, "skip_diffusion_this_step", False):
            return
        max_per_cell = element.max_per_cell
        element_type = element.element_type
        state = self._get_cached_state(element)
        # Boundaries read from Config (DIFFUSION_BOUNDARY_X_*_LEFT/RIGHT) in diffusion module, not from element.
        params = DiffusionParameters.get_or_create(
            max_per_cell, element_type,
            boundary_x_left=None,
            boundary_x_right=None,
        )
        subblock_templates = params.get_subblock_templates(
            state['p1'], state['p2'], state['p3'], state['p4'], state['p_r']
        )
        rng = self._get_thread_rng()  # thread-local so multiple elements can run in parallel without lock
        diffuse_3d_one_step_shm(
            state['read_name'], state['write_name'],
            params.n, params.max_per_cell,
            params.count_bytes, params.dirs_bytes,
            subblock_templates, params.gap_z_groups, params.bc_left, params.bc_right,
            state['p1'], state['p2'], state['p3'], state['p4'], state['p_r'],
            pool, rng
        )
        element.swap_diffusion_buffers()
        self._swap_cached_state_names(element)
    
    def _ensure_pools(self, elements):
        """Build or reuse one pool per element (outward elements get n_outward_workers each, etc.)."""
        # Preserve order: same as elements; use cached config so get_diffusion_config() is called only once per element
        ordered = []
        for e in elements:
            t = e.element_type
            ordered.append((e, self.n_outward_workers if t == 'outward' else self.n_inward_workers))

        if self.worker_pools is not None:
            # General pool manager creates/caches exactly one pool per element (order-preserving).
            elements_list = [e for e, _ in ordered]
            self._pools = self.worker_pools.get_diffusion_pools(elements_list)
            return ordered

        need = len(ordered)
        if self._pools is not None and len(self._pools) == need:
            return ordered
        if self._pools is not None:
            for p in self._pools:
                p.close()
                p.join()
        maxtasks = int(getattr(Config, "MAX_TASK_PER_CHILD", 0)) or None
        self._pools = [mp.Pool(n, maxtasksperchild=maxtasks) for _, n in ordered]
        return ordered
    
    def diffuse_multiple(self, elements):
        """
        Apply one diffusion step to all elements in parallel. Each element has its own
        process pool (subblocks run in parallel via multiprocessing), and elements are
        run concurrently via threads so all diffusion steps happen at once.
        
        Args:
            elements: list of DiffusibleElement instances (order preserved; type from config)
        """
        ordered = self._ensure_pools(elements)
        elements_list = [e for e, _ in ordered]
        pools_list = self._pools
        with ThreadPoolExecutor(max_workers=len(elements_list)) as executor:
            list(executor.map(
                lambda i: self._diffuse_with_pool(elements_list[i], pools_list[i]),
                range(len(elements_list))
            ))

    def close(self):
        """Close all cached process pools used by diffusion. (nucleation/dissolution pools are external)"""
        if self._pools is not None and not self._pools_managed_externally:
            for p in self._pools:
                p.close()
                p.join()
            self._pools = None


def prepare_diffusion_run(n, n_workers, max_per_cell, boundary_x_left, boundary_x_right, p1, p_r_extra):
    """
    Precompute z-partition (with gaps), gap_z_groups, and subblock templates.
    boundary_x_left, boundary_x_right: 'periodic'|'reflection'|'deletion' per x-side.
    Returns dict with: z_ranges, gap_z_groups, subblock_arg_templates, count_bytes, dirs_bytes, ...
    """
    bc_left = _parse_boundary(boundary_x_left)
    bc_right = _parse_boundary(boundary_x_right)
    n_blocks = max(1, n_workers)
    z_ranges, gap_z_set = _partition_domain_z(n, n_blocks)
    if not z_ranges:
        z_ranges = [(0, n - 1)]
        gap_z_set = set(range(n)) - {k for a, b in z_ranges for k in range(a, b + 1)}
    gap_z_groups = _partition_gap_z_parallel(gap_z_set, min_spacing=3, n_z=n)
    n3 = n * n * n
    count_bytes = n3 * 1
    dirs_bytes = n3 * max_per_cell * 1
    p2_val = 2 * p1
    p3_val = 3 * p1
    p4_val = 4 * p1
    p_r_val = 4 * p1 + p_r_extra
    subblock_arg_templates = [
        (n, max_per_cell, count_bytes, dirs_bytes, k_lo, k_hi, bc_left, bc_right,
         p1, p2_val, p3_val, p4_val, p_r_val)
        for (k_lo, k_hi) in z_ranges
    ]
    return {
        "z_ranges": z_ranges,
        "gap_z_groups": gap_z_groups,
        "subblock_arg_templates": subblock_arg_templates,
        "count_bytes": count_bytes,
        "dirs_bytes": dirs_bytes,
        "bc_left": bc_left,
        "bc_right": bc_right,
        "p1_val": p1,
        "p2_val": p2_val,
        "p3_val": p3_val,
        "p4_val": p4_val,
        "p_r_val": p_r_val,
        "n": n,
        "max_per_cell": max_per_cell,
    }
