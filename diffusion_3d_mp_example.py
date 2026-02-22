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
# Subblock partition: 2-cell gap between neighbouring blocks (x-axis)
# ---------------------------------------------------------------------------

def _partition_domain(n, n_blocks, bc_left, bc_right):
    """
    Partition x-axis into n_blocks interior ranges with 2-cell gaps.
    bc_left, bc_right: BC_PERIODIC (0), BC_REFLECTION (1), or BC_DELETION (2).
    - If both are PERIODIC: reserve an extra gap at x=n-1 so no worker writes across wrap.
    - Otherwise: particles stay in [0, n-1]; only (n_blocks-1)*2 gap cells between blocks.
    """
    if n_blocks <= 0 or n < 2 * n_blocks + (n_blocks - 1) * 2:
        return [], set(range(n))
    gap_width = 2
    periodic_both = (bc_left == BC_PERIODIC and bc_right == BC_PERIODIC)
    total_gaps = (n_blocks - 1) * gap_width + (1 if periodic_both else 0)
    interior_total = n - total_gaps
    if interior_total < n_blocks:
        return [], set(range(n))
    base = interior_total // n_blocks
    extra = interior_total % n_blocks
    interior_ranges = []
    x = 0
    for b in range(n_blocks):
        size = base + (1 if b < extra else 0)
        if size <= 0:
            break
        x_hi = x + size - 1
        if x_hi >= n:
            break
        if periodic_both and b == n_blocks - 1 and x_hi >= n - 1:
            x_hi = min(x_hi, n - 2)
        interior_ranges.append((x, x_hi))
        x = x_hi + 1 + gap_width
    gap_x_set = set()
    for xi in range(n):
        if not any(a <= xi <= b for a, b in interior_ranges):
            gap_x_set.add(xi)
    return interior_ranges, gap_x_set


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

# x periodic: no branch, just wrap
@numba.njit(fastmath=True, cache=True)
def _diffuse_subblock_kernel_x_periodic(
    read_count, read_dirs, write_count, write_dirs,
    x_lo, x_hi, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(n):
        for j in range(n):
            for i in range(x_lo, x_hi + 1):
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
    x_lo, x_hi, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(n):
        for j in range(n):
            for i in range(x_lo, x_hi + 1):
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
    x_lo, x_hi, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    # Coordinate convention: i=x, j=y, k=z. This kernel runs on one x-subblock only,
    # so i ranges from x_lo to x_hi (not 0..n-1). When debugging, the first i you see
    # is x_lo for this subblock; particles at x=0 are in a different subblock.
    np.random.seed(seed)
    n2 = n * n
    for k in range(n):
        for j in range(n):
            for i in range(x_lo, x_hi + 1):
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
    x_lo, x_hi, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(n):
        for j in range(n):
            for i in range(x_lo, x_hi + 1):
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
    x_lo, x_hi, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(n):
        for j in range(n):
            for i in range(x_lo, x_hi + 1):
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
    x_lo, x_hi, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(n):
        for j in range(n):
            for i in range(x_lo, x_hi + 1):
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
    x_lo, x_hi, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(n):
        for j in range(n):
            for i in range(x_lo, x_hi + 1):
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
    x_lo, x_hi, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(n):
        for j in range(n):
            for i in range(x_lo, x_hi + 1):
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
    x_lo, x_hi, n, max_per_cell, p1, p2, p3, p4, p_r, seed
):
    np.random.seed(seed)
    n2 = n * n
    for k in range(n):
        for j in range(n):
            for i in range(x_lo, x_hi + 1):
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
    bc_left, bc_right: 0=periodic, 1=reflection, 2=deletion; select one of 9 kernels.
    """
    (read_name, write_name, n, max_per_cell, count_bytes, dirs_bytes,
     x_lo, x_hi, bc_left, bc_right, p1, p2, p3, p4, p_r, seed) = args

    shm_r = shared_memory.SharedMemory(name=read_name)
    shm_w = shared_memory.SharedMemory(name=write_name)
    read_count, read_dirs = _views_from_segment(shm_r, n, max_per_cell, count_bytes, dirs_bytes)
    write_count, write_dirs = _views_from_segment(shm_w, n, max_per_cell, count_bytes, dirs_bytes)

    kernel_func = _BC_X_KERNELS_2D[bc_left][bc_right]
    kernel_func(
        read_count, read_dirs, write_count, write_dirs,
        x_lo, x_hi, n, max_per_cell,
        p1, p2, p3, p4, p_r, seed
    )

    shm_r.close()
    shm_w.close()


def _partition_gap_x_parallel(gap_x_set, min_spacing=3):
    """
    Partition gap x-coordinates into groups that can be processed in parallel.
    Gap cells at least min_spacing apart have non-overlapping writable x-ranges
    (since particles move by at most ±1, gap at x=i writes to x in {i-1, i, i+1}).
    Returns list of groups, each group is a list of x-coordinates safe to process in parallel.
    """
    if len(gap_x_set) == 0:
        return []
    gap_x_list = sorted(gap_x_set)
    groups = []
    used = set()
    for x in gap_x_list:
        if x in used:
            continue
        # Start a new group with x
        group = [x]
        used.add(x)
        # Add other gap x's that are at least min_spacing away
        for y in gap_x_list:
            if y in used:
                continue
            # Check if y is far enough from all x's already in this group
            can_add = True
            for gx in group:
                if abs(y - gx) < min_spacing:
                    can_add = False
                    break
            if can_add:
                group.append(y)
                used.add(y)
        groups.append(group)
    return groups


def _worker_gap_x(args):
    """
    Worker for gap processing: process one gap x-coordinate (all j, k for that x).
    bc_left, bc_right select one of 9 kernels (no BC branches in kernel).
    """
    (read_name, write_name, n, max_per_cell, count_bytes, dirs_bytes,
     gap_x, bc_left, bc_right, p1, p2, p3, p4, p_r, seed) = args

    shm_r = shared_memory.SharedMemory(name=read_name)
    shm_w = shared_memory.SharedMemory(name=write_name)
    read_count, read_dirs = _views_from_segment(shm_r, n, max_per_cell, count_bytes, dirs_bytes)
    write_count, write_dirs = _views_from_segment(shm_w, n, max_per_cell, count_bytes, dirs_bytes)

    kernel_func = _BC_X_KERNELS_2D[bc_left][bc_right]
    kernel_func(
        read_count, read_dirs, write_count, write_dirs,
        gap_x, gap_x, n, max_per_cell,
        p1, p2, p3, p4, p_r, seed
    )

    shm_r.close()
    shm_w.close()


def _update_gap_parallel(read_name, write_name, n, max_per_cell, count_bytes, dirs_bytes,
                        gap_groups, bc_left, bc_right, p1, p2, p3, p4, p_r, pool, rng):
    """
    Update gap cells in parallel. gap_groups is precomputed once at setup.
    """
    if len(gap_groups) == 0:
        return
    base_seed = rng.integers(0, 2**31)
    for group_idx, group in enumerate(gap_groups):
        args_list = [
            (read_name, write_name, n, max_per_cell, count_bytes, dirs_bytes,
             gap_x, bc_left, bc_right, p1, p2, p3, p4, p_r, base_seed + group_idx * 1000 + x_idx)
            for x_idx, gap_x in enumerate(group)
        ]
        pool.map(_worker_gap_x, args_list)


def diffuse_3d_one_step_shm(
    read_name, write_name, n, max_per_cell, count_bytes, dirs_bytes,
    subblock_arg_templates, gap_groups, bc_left, bc_right, p1, p2, p3, p4, p_r, pool, rng
):
    """
    One step: zero write buffer, run workers (read → write), then parallel gap.
    bc_left, bc_right: 0=periodic, 1=reflection, 2=deletion (per x-side).
    """
    # Zero write buffer
    shm_w = shared_memory.SharedMemory(name=write_name)
    write_count, _ = _views_from_segment(shm_w, n, max_per_cell, count_bytes, dirs_bytes)
    write_count.fill(0)
    shm_w.close()

    # Only (read_name, write_name) and seeds change per step
    args_list = [
        (read_name, write_name, *tpl, rng.integers(0, 2**31))
        for tpl in subblock_arg_templates
    ]
    pool.map(_worker_subblock, args_list)

    _update_gap_parallel(
        read_name, write_name, n, max_per_cell, count_bytes, dirs_bytes,
        gap_groups, bc_left, bc_right, p1, p2, p3, p4, p_r, pool, rng
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
    fallback = getattr(Config, 'DIFFUSION_BOUNDARY_X', 'periodic')
    if element_type == 'outward':
        key = f'DIFFUSION_BOUNDARY_X_OUTWARD_{side.upper()}'
    else:
        key = f'DIFFUSION_BOUNDARY_X_INWARD_{side.upper()}'
    return getattr(Config, key, fallback)


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
        self.gap_groups = prep["gap_groups"]
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
        """Get subblock argument templates with element-specific probabilities and bc_left, bc_right."""
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
    
    def __init__(self, n_outward_workers, n_inward_workers, rng):
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
        # One RNG per thread so when we run multiple elements in parallel (ThreadPoolExecutor) each has its own stream; no lock.
        self._rng_base_seed = int(rng.integers(0, 2**31))
        self._thread_local = threading.local()
        self._pools = None  # list of pools, one per element; created on first diffuse_multiple
        # Config (max_per_cell, element_type) is constant per element; boundaries come from Config in diffusion module.
        self._config_cache = {}  # id(element) -> config dict, filled once per element
        self._state_cache = {}   # id(element) -> state dict (read_name, write_name, p1..p_r); names updated after each swap
    
    # def _get_cached_config(self, element):
    #     """Return diffusion config for element. max_per_cell from Config (not element); element_type from element once, then cached."""
    #     eid = id(element)
    #     if eid not in self._config_cache:
    #         # Only ask element for element_type (outward/inward); max_per_cell comes from Config
    #         element_type = element.get_diffusion_config().get('element_type', 'outward')
    #         if element_type == 'outward':
    #             max_per_cell = getattr(Config, 'OUTWARD_DIFFUSION_MAX_PER_CELL', getattr(Config, 'DIFFUSION_MAX_PER_CELL_FALLBACK', 50))
    #         else:
    #             max_per_cell = getattr(Config, 'INWARD_DIFFUSION_MAX_PER_CELL', getattr(Config, 'DIFFUSION_MAX_PER_CELL_FALLBACK', 50))
    #         self._config_cache[eid] = {'element_type': element_type, 'max_per_cell': max_per_cell}
    #     return self._config_cache[eid]
    
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

    # def register_elements(self, elements):
    #     """
    #     Register elements that will be diffused. Call once at init so the engine knows each
    #     element's max_per_cell (and element_type) and pre-creates DiffusionParameters.
    #     After this, diffuse() / diffuse_multiple() use the cached config per element.
        
    #     Args:
    #         elements: iterable of DiffusibleElement instances (e.g. [active_elem, oxidant_elem])
    #     """
    #     for elem in elements:
    #         self._get_cached_state(elem)  # prime state cache (read/write names, p1..p_r)
    #         max_per_cell = elem.max_per_cell
    #         element_type = elem.element_type
    #         DiffusionParameters.get_or_create(
    #             max_per_cell, element_type,
    #             boundary_x_left=None,
    #             boundary_x_right=None,
    #         )
    
    def _diffuse_with_pool(self, element, pool):
        """Apply one diffusion step to an element using the given dedicated pool."""
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
            subblock_templates, params.gap_groups, params.bc_left, params.bc_right,
            state['p1'], state['p2'], state['p3'], state['p4'], state['p_r'],
            pool, rng
        )
        element.swap_diffusion_buffers()
        self._swap_cached_state_names(element)
    
    # def diffuse(self, element):
    #     """
    #     Apply one Chopard-Droz diffusion step to an element.
    #     Uses a temporary pool (one per element type) for single-element use.
        
    #     Args:
    #         element: DiffusibleElement instance that implements the protocol
    #     """
    #     config = self._get_cached_config(element)
    #     element_type = config.get('element_type', 'outward')
    #     n = self.n_inward_workers if element_type == 'inward' else self.n_outward_workers
    #     pool = mp.Pool(n)
    #     try:
    #         self._diffuse_with_pool(element, pool)
    #     finally:
    #         pool.close()
    #         pool.join()
    
    def _ensure_pools(self, elements):
        """Build or reuse one pool per element (outward elements get n_outward_workers each, etc.)."""
        # Preserve order: same as elements; use cached config so get_diffusion_config() is called only once per element
        ordered = []
        for e in elements:
            t = e.element_type
            ordered.append((e, self.n_outward_workers if t == 'outward' else self.n_inward_workers))
        need = len(ordered)
        if self._pools is not None and len(self._pools) == need:
            return ordered
        if self._pools is not None:
            for p in self._pools:
                p.close()
                p.join()
        self._pools = [mp.Pool(n) for _, n in ordered]
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
        """Close all cached process pools. Call when done with the engine."""
        if self._pools is not None:
            for p in self._pools:
                p.close()
                p.join()
            self._pools = None


def prepare_diffusion_run(n, n_workers, max_per_cell, boundary_x_left, boundary_x_right, p1, p_r_extra):
    """
    Precompute partition, gap groups, subblock templates and p values.
    boundary_x_left, boundary_x_right: 'periodic'|'reflection'|'deletion' per x-side.
    Returns dict with: interior_ranges, gap_groups, subblock_arg_templates, count_bytes, dirs_bytes,
    bc_left, bc_right, p1_val, ...
    """
    bc_left = _parse_boundary(boundary_x_left)
    bc_right = _parse_boundary(boundary_x_right)
    n_blocks = max(1, n_workers)
    interior_ranges, gap_x_set = _partition_domain(n, n_blocks, bc_left, bc_right)
    if not interior_ranges:
        interior_ranges = [(0, min(n - 2, n - 1))]
        gap_x_set = set(range(n)) - {x for a, b in interior_ranges for x in range(a, b + 1)}
    gap_groups = _partition_gap_x_parallel(gap_x_set, min_spacing=3)
    n3 = n * n * n
    count_bytes = n3 * 1
    dirs_bytes = n3 * max_per_cell * 1
    p2_val = 2 * p1
    p3_val = 3 * p1
    p4_val = 4 * p1
    p_r_val = 4 * p1 + p_r_extra
    subblock_arg_templates = [
        (n, max_per_cell, count_bytes, dirs_bytes, x_lo, x_hi, bc_left, bc_right,
         p1, p2_val, p3_val, p4_val, p_r_val)
        for (x_lo, x_hi) in interior_ranges
    ]
    return {
        "interior_ranges": interior_ranges,
        "gap_groups": gap_groups,
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


def run_example():
    n = 300
    n_workers = 7
    n_blocks = n_workers
    n_steps = 100
    total_particles = 100_000
    max_per_cell = 2

    # Boundary condition per x-side: "periodic", "reflection", or "deletion"
    boundary_x_left = "periodic"
    boundary_x_right = "periodic"
    bc_left = _parse_boundary(boundary_x_left)
    bc_right = _parse_boundary(boundary_x_right)

    interior_ranges, gap_x_set = _partition_domain(n, n_blocks, bc_left, bc_right)
    if not interior_ranges:
        print("Partition failed. Using single block.")
        interior_ranges = [(0, min(n - 2, n - 1))]
        gap_x_set = set(range(n)) - {x for a, b in interior_ranges for x in range(a, b + 1)}
    gap_groups = _partition_gap_x_parallel(gap_x_set, min_spacing=3)

    n3 = n * n * n
    count_dtype = np.int8  # max 127; max_per_cell ≤ 50
    count_bytes = n3 * np.dtype(count_dtype).itemsize
    # Packed dirs: one byte per (dx,dy,dz) with values in {-1,0,1}
    dirs_bytes = n3 * max_per_cell * 1
    segment_bytes = count_bytes + dirs_bytes

    # Two contiguous shared segments (one per buffer): [count][dirs]
    shm_A = shared_memory.SharedMemory(create=True, size=segment_bytes, name=None)
    shm_B = shared_memory.SharedMemory(create=True, size=segment_bytes, name=None)

    A_count, A_dirs = _views_from_segment(shm_A, n, max_per_cell, count_bytes, dirs_bytes)
    B_count, B_dirs = _views_from_segment(shm_B, n, max_per_cell, count_bytes, dirs_bytes)

    A_count.fill(0)
    A_dirs.fill(0)
    rng = np.random.default_rng(42)
    for _ in range(total_particles):
        i, j, k = rng.integers(0, n, size=3)
        idx = _idx(i, j, k, n)
        c = A_count[idx]
        if c < max_per_cell:
            A_dirs[idx, c] = _DIRS_6_PACKED[rng.integers(0, 6)]
            A_count[idx] = c + 1

    class FakePRanges:
        pass
    p_ranges = FakePRanges()
    p1, p_r_extra = 0.15, 0.1
    p_ranges.p1_range = p1
    p_ranges.p2_range = 2 * p1
    p_ranges.p3_range = 3 * p1
    p_ranges.p4_range = 4 * p1
    p_ranges.p_r_range = 4 * p1 + p_r_extra

    # Precompute once: subblock arg templates with bc_left, bc_right
    p1_val = p_ranges.p1_range
    p2_val = p_ranges.p2_range
    p3_val = p_ranges.p3_range
    p4_val = p_ranges.p4_range
    p_r_val = p_ranges.p_r_range
    subblock_arg_templates = [
        (n, max_per_cell, count_bytes, dirs_bytes, x_lo, x_hi, bc_left, bc_right,
         p1_val, p2_val, p3_val, p4_val, p_r_val)
        for (x_lo, x_hi) in interior_ranges
    ]

    read_name, write_name = shm_A.name, shm_B.name

    try:
        t0 = time.perf_counter()
        with mp.Pool(n_workers) as pool:
            for step in range(n_steps):
                print(f"Step {step}")
                diffuse_3d_one_step_shm(
                    read_name, write_name, n, max_per_cell, count_bytes, dirs_bytes,
                    subblock_arg_templates, gap_groups, bc_left, bc_right,
                    p1_val, p2_val, p3_val, p4_val, p_r_val, pool, rng
                )
                read_name, write_name = write_name, read_name
        elapsed = time.perf_counter() - t0

        if n_steps % 2 == 0:
            n_final = int(B_count.sum())
        else:
            n_final = int(A_count.sum())
        print(f"  Done. Total particles: {n_final}, time: {elapsed:.3f}s, steps/s: {n_steps/elapsed:.2f}")
    finally:
        shm_A.close()
        shm_B.close()
        shm_A.unlink()
        shm_B.unlink()

    return None


if __name__ == "__main__":
    run_example()
