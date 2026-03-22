from multiprocessing import shared_memory
import numpy as np
from utils.numba_functions import (
    dissolution_subblock_kernel_snapshot,
    dissolution_subblock_kernel_snapshot_with_blocks,

)
from .neigh_indexes import (
    OFFSETS_26,
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


def dissolution_subblock_worker(task):
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
