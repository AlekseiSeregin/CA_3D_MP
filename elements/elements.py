from utils.numba_functions import *
from configuration import Config
from multiprocessing import shared_memory
from cellular_automata.nes_for_mp import *
import sys
import random
import numpy as np


try:
    from diffusion_3d_mp_example import (
        DiffusionEngine,
        _DIRS_6_PACKED,
        _views_from_segment,  # Helper for buffer creation
    )
    _DIFFUSION_SHM_AVAILABLE = True
except ImportError:
    _DIFFUSION_SHM_AVAILABLE = False
    DiffusionEngine = None
    DiffusibleElement = None
    _DIRS_6_PACKED = None


def _diffusion_grid_idx(i, j, k, n):
    """Flat index for 3D (i, j, k) in C order: i changes fastest."""
    ni, nj, nk, nn = int(i), int(j), int(k), int(n)
    return ni + nn * (nj + nn * nk)


def flat_to_grid_sync(count, dirs_grid, cells_flat, dirs_flat, n, max_per_cell):
    """
    Fill grid (count, dirs_grid) from flat representation.
    cells_flat: (3, N) int (i, j, k) per column; dirs_flat: (3, N) int in {-1,0,1}.
    Clamps to grid bounds and drops particles beyond max_per_cell per cell.
    Part of element/setup logic; kept for any code that still needs to fill grid from flat data.
    """
    n3 = n * n * n
    count.fill(0)
    dirs_grid.fill(0)
    for p in range(cells_flat.shape[1]):
        i, j, k = int(cells_flat[0, p]), int(cells_flat[1, p]), int(cells_flat[2, p])
        if i < 0 or i >= n or j < 0 or j >= n or k < 0 or k >= n:
            continue
        idx = _diffusion_grid_idx(i, j, k, n)
        c = count[idx]
        if c >= max_per_cell:
            continue
        dx, dy, dz = int(dirs_flat[0, p]), int(dirs_flat[1, p]), int(dirs_flat[2, p])
        packed = (dx + 1) + (dy + 1) * 4 + (dz + 1) * 16
        dirs_grid[idx, c] = np.uint8(min(max(packed, 0), 255))
        count[idx] = c + 1


# ---------------------------------------------------------------------------
# Buffer initialization utilities (moved from diffusion module)
# ---------------------------------------------------------------------------

def create_diffusion_buffers(n, max_per_cell):
    """
    Create two shared-memory segments (ping-pong) for count + packed dirs.
    Returns (shm_A, shm_B, A_count, A_dirs, B_count, B_dirs, count_bytes, dirs_bytes).
    Caller must close/unlink shm when done.
    
    This is initialization logic, not part of diffusion operations.
    DiffusionEngine only operates on already-initialized buffers.
    """
    n3 = n * n * n
    count_dtype = np.int8
    count_bytes = n3 * np.dtype(count_dtype).itemsize
    dirs_bytes = n3 * max_per_cell * 1
    segment_bytes = count_bytes + dirs_bytes
    shm_A = shared_memory.SharedMemory(create=True, size=segment_bytes, name=None)
    shm_B = shared_memory.SharedMemory(create=True, size=segment_bytes, name=None)
    A_count, A_dirs = _views_from_segment(shm_A, n, max_per_cell, count_bytes, dirs_bytes)
    B_count, B_dirs = _views_from_segment(shm_B, n, max_per_cell, count_bytes, dirs_bytes)
    A_count.fill(0)
    A_dirs.fill(0)
    B_count.fill(0)
    B_dirs.fill(0)
    return shm_A, shm_B, A_count, A_dirs, B_count, B_dirs


class ActiveElem:
    def __init__(self, settings):
        self.elem_name = settings.ELEMENT
        self.cells_per_axis = Config.N_CELLS_PER_AXIS
        self.neigh_range = Config.NEIGH_RANGE
        self.shape = (self.cells_per_axis, self.cells_per_axis, self.cells_per_axis)
        self.p1_range = settings.PROBABILITIES[0]
        self.p2_range = 2 * self.p1_range
        self.p3_range = 3 * self.p1_range
        self.p4_range = 4 * self.p1_range
        self.p_r_range = self.p4_range + settings.PROBABILITIES[1]
        self.n_per_page = settings.N_PER_PAGE

        self.p_ranges = PRanges(self.p1_range, self.p2_range, self.p3_range, self.p4_range, self.p_r_range)
        self.p_ranges_scale = PRanges(self.p1_range, self.p2_range, self.p3_range, self.p4_range, self.p_r_range)

        self.precip_transform_depth = int(Config.PRECIP_TRANSFORM_DEPTH)

        extended_axis = self.cells_per_axis + self.neigh_range
        self.extended_shape = (self.cells_per_axis, self.cells_per_axis, extended_axis)

        self.diffuse = None  # must be defined elsewhere
        self.scale = None  # must be defined elsewhere

        self.current_count = None
        self.shms_unlinked = False

        # 3D diffusion grid (count + packed dirs) is now PRIMARY storage (no flat arrays)
        self._diff_max_per_cell = getattr(Config, 'DIFFUSION_MAX_PER_CELL', 50)
        self._diff_shm_A = self._diff_shm_B = None
        self._diff_A_count = self._diff_A_dirs = self._diff_B_count = self._diff_B_dirs = None
        self._diff_read_name = self._diff_write_name = None
        self._diff_step_args = None
        
        self._init_diffusion_buffers()
        self._init_particles_in_grid(settings)

    def _init_diffusion_buffers(self):
        """Create shared-memory diffusion grids (count + dirs) and prepare run."""
        n = self.cells_per_axis
        max_per_cell = self._diff_max_per_cell
        
        # Create element-specific buffers
        # Note: DiffusionParameters are now handled internally by DiffusionEngine
        (self._diff_shm_A, self._diff_shm_B,
         self._diff_A_count, self._diff_A_dirs,
         self._diff_B_count, self._diff_B_dirs) = create_diffusion_buffers(n, max_per_cell)
        self._diff_read_name = self._diff_shm_A.name
        self._diff_write_name = self._diff_shm_B.name
        
        # Store only element-specific probability values
        p1 = self.p1_range
        p_r_extra = self.p_r_range - self.p4_range
        self._diff_step_args = {
            "p1": p1,
            "p2": 2 * p1,
            "p3": 3 * p1,
            "p4": 4 * p1,
            "p_r": 4 * p1 + p_r_extra,
        }

    def _init_particles_in_grid(self, settings):
        """Initialize particles directly in the 3D grid (no flat arrays). CONC_PRECISION: 'rand' or 'exact'. SPACE_FILL: 'full' or 'half'."""
        n = self.cells_per_axis
        max_per_cell = self._diff_max_per_cell
        rng = np.random.default_rng()
        count = self._diff_A_count
        dirs = self._diff_A_dirs
        count.fill(0)
        dirs.fill(0)
        n2 = n * n
        k_lo = n // 2 if getattr(settings, 'SPACE_FILL', 'full').lower() == 'half' else 0
        # Packed directions: (dx+1)+(dy+1)*4+(dz+1)*16 for ±x, ±y, ±z (same order as diffusion module)
        packed_dirs = _DIRS_6_PACKED if _DIRS_6_PACKED is not None else np.array([20, 22, 17, 25, 5, 37], dtype=np.uint8)

        if settings.CONC_PRECISION.lower() == 'rand':
            total_particles = int(self.n_per_page * self.cells_per_axis)
            for _ in range(total_particles):
                k = rng.integers(k_lo, n)
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                idx = i + n * j + n2 * k
                if count[idx] < max_per_cell:
                    count[idx] += 1
                    dirs[idx, count[idx] - 1] = packed_dirs[rng.integers(0, 6)]
        elif settings.CONC_PRECISION.lower() == 'exact':
            n_per = min(int(self.n_per_page), n * n)
            for k in range(k_lo, n):
                indices_2d = rng.choice(n * n, size=n_per, replace=False)
                for idx_2d in indices_2d:
                    i = idx_2d % n
                    j = idx_2d // n
                    idx = i + n * j + n2 * k
                    if count[idx] < max_per_cell:
                        count[idx] += 1
                        dirs[idx, count[idx] - 1] = packed_dirs[rng.integers(0, 6)]
        else:
            raise ValueError(f"Wrong CONC_PRECISION for outward element! (use 'exact' or 'rand')")

    def _get_current_grid(self):
        """Get current read buffer (count, dirs) from grid."""
        read_count = self._diff_A_count if self._diff_read_name == self._diff_shm_A.name else self._diff_B_count
        read_dirs = self._diff_A_dirs if self._diff_read_name == self._diff_shm_A.name else self._diff_B_dirs
        return read_count, read_dirs

    def get_diffusion_state(self):
        """Return current diffusion state for DiffusionEngine (DiffusibleElement protocol)."""
        args = self._diff_step_args
        return {
            'read_name': self._diff_read_name,
            'write_name': self._diff_write_name,
            'p1': args["p1"],
            'p2': args["p2"],
            'p3': args["p3"],
            'p4': args["p4"],
            'p_r': args["p_r"],
        }
    
    def get_diffusion_config(self):
        """Return diffusion configuration (DiffusibleElement protocol)."""
        p1 = self.p1_range
        p_r_extra = self.p_r_range - self.p4_range
        return {
            'max_per_cell': self._diff_max_per_cell,
            'element_type': 'outward',  # ActiveElem is outward diffusion
            'p1': p1,
            'p_r_extra': p_r_extra,
        }
    
    def swap_diffusion_buffers(self):
        """Swap read/write buffers after diffusion step (DiffusibleElement protocol)."""
        self._diff_read_name, self._diff_write_name = self._diff_write_name, self._diff_read_name

    def get_diffusion_grid_3d(self, copy=False):
        """
        Return the current diffusion read buffer as 3D arrays.
        
        Returns:
            count_3d: (n, n, n) int8 – particle count per cell
            dirs_3d: (n, n, n, max_per_cell) uint8 – packed direction per cell/slot
        If copy=True returns copies; otherwise returns views (same memory as shared buffer).
        """
        read_count, read_dirs = self._get_current_grid()
        n = self.cells_per_axis
        max_per_cell = self._diff_max_per_cell
        count_3d = read_count.reshape(n, n, n)
        dirs_3d = read_dirs.reshape(n, n, n, max_per_cell)
        if copy:
            return count_3d.copy(), dirs_3d.copy()
        return count_3d, dirs_3d
    
    def close_and_unlink_shm(self):
        if not self.shms_unlinked:
            # c3d_shared removed - no longer needed
            if _DIFFUSION_SHM_AVAILABLE and self._diff_shm_A is not None:
                self._diff_shm_A.close()
                self._diff_shm_A.unlink()
                self._diff_shm_B.close()
                self._diff_shm_B.unlink()
            self.shms_unlinked = True


class OxidantElem:
    def __init__(self, settings, utils):
        self.elem_name = settings.ELEMENT
        self.cells_per_axis = Config.N_CELLS_PER_AXIS
        self.p1_range = settings.PROBABILITIES[0]
        self.p2_range = 2 * self.p1_range
        self.p3_range = 3 * self.p1_range
        self.p4_range = 4 * self.p1_range
        self.p_r_range = self.p4_range + settings.PROBABILITIES[1]
        self.p0_2d = settings.PROBABILITIES_2D
        self.n_per_page = settings.N_PER_PAGE
        self.neigh_range = Config.NEIGH_RANGE
        self.current_count = 0
        self.furthest_index = None

        self.p_ranges_scale = self.generate_prob_ranges(settings.PROBABILITIES_SCALE)
        self.p_ranges_interface = self.generate_prob_ranges(settings.PROBABILITIES_INTERFACE)

        self.extended_axis = self.cells_per_axis + self.neigh_range
        self.extended_shape = (self.cells_per_axis, self.cells_per_axis, self.extended_axis)

        self.shms_unlinked = False

        self.scale = None
        self.diffuse = None
        self.n_boost_steps = Config.N_BOOST_STEPS

        self.utils = utils
        self.microstructure = None

        # 3D diffusion grid (inward) is now PRIMARY storage (no flat arrays)
        self._diff_max_per_cell = getattr(Config, 'DIFFUSION_MAX_PER_CELL', 50)
        self._diff_shm_A = self._diff_shm_B = None
        self._diff_A_count = self._diff_A_dirs = self._diff_B_count = self._diff_B_dirs = None
        self._diff_read_name = self._diff_write_name = None
        self._diff_step_args = None
        
        self._init_diffusion_buffers_oxidant()
        # Initialize with empty grid (fill_first_page will add particles)
        self.current_count = 0
        self.fill_first_page()

        # self.microstructure = voronoi.VoronoiMicrostructure(self.cells_per_axis)
        # self.microstructure.generate_voronoi_3d(50, seeds="own")
        # self.microstructure.show_microstructure(self.cells_per_axis)
        # self.cross_shifts = np.array([[1, 0, 0], [0, 1, 0],
        #                               [-1, 0, 0], [0, -1, 0],
        #                               [0, 0, -1]], dtype=np.byte)

    def diffuse_bulk(self):
        """
        DEPRECATED: Legacy method using flat arrays.
        Use DiffusionEngine.diffuse(element) instead - operates directly on grid.
        """
        raise NotImplementedError("diffuse_bulk() removed - use DiffusionEngine.diffuse(element) instead")
        if cells_flat.shape[1] == 0:
            return
        
        randomise = np.array(np.random.random_sample(cells_flat.shape[1]), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        dirs_flat[:, temp_ind] = np.roll(dirs_flat[:, temp_ind], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        dirs_flat[:, temp_ind] = np.roll(dirs_flat[:, temp_ind], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        dirs_flat[:, temp_ind] = np.roll(dirs_flat[:, temp_ind], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        dirs_flat[:, temp_ind] = np.roll(dirs_flat[:, temp_ind], 2, axis=0)
        dirs_flat[:, temp_ind] *= -1
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        dirs_flat[:, temp_ind] *= -1
        cells_flat = np.add(cells_flat, dirs_flat, casting="unsafe")
        
        # Adjust coordinates for boundary conditions
        ind = np.where(cells_flat[2] < 0)[0]
        # open left bound
        keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
        keep_mask[ind] = False
        cells_flat = cells_flat[:, keep_mask]
        dirs_flat = dirs_flat[:, keep_mask]
        
        cells_flat[0, np.where(cells_flat[0] <= -1)] = self.cells_per_axis - 1
        cells_flat[0, np.where(cells_flat[0] >= self.cells_per_axis)] = 0
        cells_flat[1, np.where(cells_flat[1] <= -1)] = self.cells_per_axis - 1
        cells_flat[1, np.where(cells_flat[1] >= self.cells_per_axis)] = 0
        ind = np.where(cells_flat[2] >= self.cells_per_axis)[0]
        # open right bound
        keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
        keep_mask[ind] = False
        cells_flat = cells_flat[:, keep_mask]
        dirs_flat = dirs_flat[:, keep_mask]
        
        self._set_flat_cells_dirs(cells_flat, dirs_flat)
        self.current_count = len(np.where(cells_flat[2] == 0)[0]) if cells_flat.shape[1] > 0 else 0
        self.fill_first_page()

    def diffuse_gb(self):
        """
        DEPRECATED: Legacy method using flat arrays.
        Use DiffusionEngine.diffuse(element) instead - operates directly on grid.
        """
        raise NotImplementedError("diffuse_gb() removed - use DiffusionEngine.diffuse(element) instead")
        if cells_flat.shape[1] == 0:
            return
        
        # Diffusion along grain boundaries
        exists = self.microstructure.grain_boundaries[cells_flat[0], cells_flat[1], cells_flat[2]]
        t_ind_in_gb, ind_out_gb = separate_in_gb(exists)

        randomise = np.array(np.random.random_sample(len(t_ind_in_gb)), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p0_2d)[0], dtype=np.uint32)

        ind_in_gb = t_ind_in_gb[temp_ind]
        temp_ = np.delete(t_ind_in_gb, temp_ind)
        ind_out_gb = np.concatenate((ind_out_gb, temp_))
        in_gb = np.array(cells_flat[:, ind_in_gb], dtype=np.short)

        boost_vector = np.array(self.microstructure.jump_directions[in_gb[0], in_gb[1], in_gb[2]],
                                dtype=np.short).transpose()
        cells_flat[:, ind_in_gb] += boost_vector

        # Diffusion in bulk
        randomise = np.array(np.random.random_sample(len(ind_out_gb)), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        dirs_flat[:, ind_out_gb[temp_ind]] = np.roll(dirs_flat[:, ind_out_gb[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        dirs_flat[:, ind_out_gb[temp_ind]] = np.roll(dirs_flat[:, ind_out_gb[temp_ind]], 1, axis=0)
        dirs_flat[:, ind_out_gb[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        dirs_flat[:, ind_out_gb[temp_ind]] = np.roll(dirs_flat[:, ind_out_gb[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        dirs_flat[:, ind_out_gb[temp_ind]] = np.roll(dirs_flat[:, ind_out_gb[temp_ind]], 2, axis=0)
        dirs_flat[:, ind_out_gb[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        dirs_flat[:, ind_out_gb[temp_ind]] *= -1

        cells_flat = np.add(cells_flat, dirs_flat, casting="unsafe")
        # Adjust coordinates for boundary conditions
        ind = np.where(cells_flat[2] < 0)[0]
        # open left bound
        keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
        keep_mask[ind] = False
        cells_flat = cells_flat[:, keep_mask]
        dirs_flat = dirs_flat[:, keep_mask]

        cells_flat[0, np.where(cells_flat[0] <= -1)] = self.cells_per_axis - 1
        cells_flat[0, np.where(cells_flat[0] >= self.cells_per_axis)] = 0
        cells_flat[1, np.where(cells_flat[1] <= -1)] = self.cells_per_axis - 1
        cells_flat[1, np.where(cells_flat[1] >= self.cells_per_axis)] = 0

        ind = np.where(cells_flat[2] >= self.cells_per_axis)[0]
        # open right bound
        keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
        keep_mask[ind] = False
        cells_flat = cells_flat[:, keep_mask]
        dirs_flat = dirs_flat[:, keep_mask]

        self._set_flat_cells_dirs(cells_flat, dirs_flat)
        self.current_count = len(np.where(cells_flat[2] == 0)[0]) if cells_flat.shape[1] > 0 else 0
        self.fill_first_page()

    def diffuse_with_scale(self):
        """
        DEPRECATED: Legacy method using flat arrays.
        Use DiffusionEngine.diffuse(element) instead - operates directly on grid.
        """
        raise NotImplementedError("diffuse_with_scale() removed - use DiffusionEngine.diffuse(element) instead")
        if cells_flat.shape[1] == 0:
            return
        
        # Diffusion at the interface between matrix the scale
        self.diffuse_interface()

        # Diffusion through the scale. If the current particle is inside the product particle it will be reflected
        out_scale = check_in_scale(self.scale, cells_flat, dirs_flat)

        # Mixing particles according to Chopard and Droz
        randomise = np.array(np.random.random_sample(out_scale.size), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        dirs_flat[:, out_scale[temp_ind]] = np.roll(dirs_flat[:, out_scale[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        dirs_flat[:, out_scale[temp_ind]] = np.roll(dirs_flat[:, out_scale[temp_ind]], 1, axis=0)
        dirs_flat[:, out_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        dirs_flat[:, out_scale[temp_ind]] = np.roll(dirs_flat[:, out_scale[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        dirs_flat[:, out_scale[temp_ind]] = np.roll(dirs_flat[:, out_scale[temp_ind]], 2, axis=0)
        dirs_flat[:, out_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        dirs_flat[:, out_scale[temp_ind]] *= -1

        cells_flat = np.add(cells_flat, dirs_flat, casting="unsafe")
        # Adjust coordinates for boundary conditions
        ind = np.where(cells_flat[2] < 0)[0]
        # open left bound
        keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
        keep_mask[ind] = False
        cells_flat = cells_flat[:, keep_mask]
        dirs_flat = dirs_flat[:, keep_mask]

        cells_flat[0, np.where(cells_flat[0] <= -1)] = self.cells_per_axis - 1
        cells_flat[0, np.where(cells_flat[0] >= self.cells_per_axis)] = 0
        cells_flat[1, np.where(cells_flat[1] <= -1)] = self.cells_per_axis - 1
        cells_flat[1, np.where(cells_flat[1] >= self.cells_per_axis)] = 0

        ind = np.where(cells_flat[2] >= self.cells_per_axis)[0]
        # open right bound
        keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
        keep_mask[ind] = False
        cells_flat = cells_flat[:, keep_mask]
        dirs_flat = dirs_flat[:, keep_mask]

        self._set_flat_cells_dirs(cells_flat, dirs_flat)
        self.current_count = len(np.where(cells_flat[2] == 0)[0]) if cells_flat.shape[1] > 0 else 0
        self.fill_first_page()

    def diffuse_with_scale_adj(self, time=0):
        """
        DEPRECATED: Legacy method using flat arrays.
        Use DiffusionEngine.diffuse(element) instead - operates directly on grid.
        """
        raise NotImplementedError("diffuse_with_scale_adj() removed - use DiffusionEngine.diffuse(element) instead")
        if cells_flat.shape[1] == 0:
            return
        
        # Diffusion through the scale. If the current particle is inside the product particle it will be reflected
        out_scale, in_scale = check_in_scale_adj(self.scale, cells_flat)

        # Mixing particles according to Chopard and Droz (out of scale)
        randomise = np.array(np.random.random_sample(out_scale.size), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        dirs_flat[:, out_scale[temp_ind]] = np.roll(dirs_flat[:, out_scale[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        dirs_flat[:, out_scale[temp_ind]] = np.roll(dirs_flat[:, out_scale[temp_ind]], 1, axis=0)
        dirs_flat[:, out_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        dirs_flat[:, out_scale[temp_ind]] = np.roll(dirs_flat[:, out_scale[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        dirs_flat[:, out_scale[temp_ind]] = np.roll(dirs_flat[:, out_scale[temp_ind]], 2, axis=0)
        dirs_flat[:, out_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        dirs_flat[:, out_scale[temp_ind]] *= -1

        # IN Scale Diffusion
        randomise = np.array(np.random.random_sample(in_scale.size), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p_ranges_scale.p1_range)[0], dtype=np.uint32)
        dirs_flat[:, in_scale[temp_ind]] = np.roll(dirs_flat[:, in_scale[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p1_range) &
                                     (randomise <= self.p_ranges_scale.p2_range))[0], dtype=np.uint32)
        dirs_flat[:, in_scale[temp_ind]] = np.roll(dirs_flat[:, in_scale[temp_ind]], 1, axis=0)
        dirs_flat[:, in_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p2_range) &
                                     (randomise <= self.p_ranges_scale.p3_range))[0], dtype=np.uint32)
        dirs_flat[:, in_scale[temp_ind]] = np.roll(dirs_flat[:, in_scale[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p3_range) &
                                     (randomise <= self.p_ranges_scale.p4_range))[0], dtype=np.uint32)
        dirs_flat[:, in_scale[temp_ind]] = np.roll(dirs_flat[:, in_scale[temp_ind]], 2, axis=0)
        dirs_flat[:, in_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p4_range) &
                                     (randomise <= self.p_ranges_scale.p_r_range))[0], dtype=np.uint32)
        dirs_flat[:, in_scale[temp_ind]] *= -1

        cells_flat = np.add(cells_flat, dirs_flat, casting="unsafe")
        # Adjust coordinates for boundary conditions
        ind = np.where(cells_flat[2] < 0)[0]
        # open left bound
        keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
        keep_mask[ind] = False
        cells_flat = cells_flat[:, keep_mask]
        dirs_flat = dirs_flat[:, keep_mask]

        cells_flat[0, np.where(cells_flat[0] <= -1)] = self.cells_per_axis - 1
        cells_flat[0, np.where(cells_flat[0] >= self.cells_per_axis)] = 0
        cells_flat[1, np.where(cells_flat[1] <= -1)] = self.cells_per_axis - 1
        cells_flat[1, np.where(cells_flat[1] >= self.cells_per_axis)] = 0

        ind = np.where(cells_flat[2] >= self.cells_per_axis)[0]
        # open right bound
        keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
        keep_mask[ind] = False
        cells_flat = cells_flat[:, keep_mask]
        dirs_flat = dirs_flat[:, keep_mask]

        self._set_flat_cells_dirs(cells_flat, dirs_flat)
        self.current_count = len(np.where(cells_flat[2] == 0)[0]) if cells_flat.shape[1] > 0 else 0
        self.fill_first_page(time=time)

    def diffuse_interface(self):
        """
        DEPRECATED: Legacy method using flat arrays.
        Use DiffusionEngine.diffuse(element) instead - operates directly on grid.
        """
        raise NotImplementedError("diffuse_interface() removed - use DiffusionEngine.diffuse(element) instead")
        if cells_flat.shape[1] == 0:
            return
        
        all_arounds = self.utils.calc_sur_ind_interface(cells_flat, dirs_flat, self.extended_axis - 1)
        neighbours = go_around_bool(self.scale, all_arounds)
        to_boost = np.array([sum(n_arr[:-1]) * (not n_arr[-1]) for n_arr in neighbours])
        to_boost = np.array(np.where(to_boost)[0])

        if len(to_boost) > 0:
            for _ in range(self.n_boost_steps):
                cells_flat[:, to_boost] = np.add(cells_flat[:, to_boost], dirs_flat[:, to_boost], casting="unsafe")
            # Adjust coordinates for boundary conditions
            cells_flat[0, to_boost[np.where(cells_flat[0, to_boost] <= -1)]] = self.cells_per_axis - 1
            cells_flat[0, to_boost[np.where(cells_flat[0, to_boost] >= self.cells_per_axis)]] = 0
            cells_flat[1, to_boost[np.where(cells_flat[1, to_boost] <= -1)]] = self.cells_per_axis - 1
            cells_flat[1, to_boost[np.where(cells_flat[1, to_boost] >= self.cells_per_axis)]] = 0

            ind = np.where(cells_flat[2, to_boost] < 0)[0]
            # open left bound
            keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
            keep_mask[to_boost[ind]] = False
            cells_flat = cells_flat[:, keep_mask]
            dirs_flat = dirs_flat[:, keep_mask]

            ind = np.where(cells_flat[2] >= self.cells_per_axis)[0]
            # open right bound
            keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
            keep_mask[ind] = False
            cells_flat = cells_flat[:, keep_mask]
            dirs_flat = dirs_flat[:, keep_mask]
            
            self._set_flat_cells_dirs(cells_flat, dirs_flat)

    def diffuse_interface_adj(self):
        """
        DEPRECATED: Legacy method using flat arrays.
        Use DiffusionEngine.diffuse(element) instead - operates directly on grid.
        """
        raise NotImplementedError("diffuse_interface_adj() removed - use DiffusionEngine.diffuse(element) instead")
        if cells_flat.shape[1] == 0:
            return
        
        all_arounds = self.utils.calc_sur_ind_interface_adj(cells_flat, dirs_flat, self.extended_axis - 1)
        in_int, blocked, out_int = separate_in_interface(self.scale, all_arounds)

        # Mixing particles according to Chopard and Droz (out of interface)
        randomise = np.array(np.random.random_sample(out_int.size), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        dirs_flat[:, out_int[temp_ind]] = np.roll(dirs_flat[:, out_int[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        dirs_flat[:, out_int[temp_ind]] = np.roll(dirs_flat[:, out_int[temp_ind]], 1, axis=0)
        dirs_flat[:, out_int[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        dirs_flat[:, out_int[temp_ind]] = np.roll(dirs_flat[:, out_int[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        dirs_flat[:, out_int[temp_ind]] = np.roll(dirs_flat[:, out_int[temp_ind]], 2, axis=0)
        dirs_flat[:, out_int[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        dirs_flat[:, out_int[temp_ind]] *= -1

        # INTERFACE Diffusion
        randomise = np.array(np.random.random_sample(in_int.size), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p_ranges_interface.p1_range)[0], dtype=np.uint32)
        dirs_flat[:, in_int[temp_ind]] = np.roll(dirs_flat[:, in_int[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p_ranges_interface.p1_range) & (randomise <= self.p_ranges_interface.p2_range))[0], dtype=np.uint32)
        dirs_flat[:, in_int[temp_ind]] = np.roll(dirs_flat[:, in_int[temp_ind]], 1, axis=0)
        dirs_flat[:, in_int[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p_ranges_interface.p2_range) & (randomise <= self.p_ranges_interface.p3_range))[0], dtype=np.uint32)
        dirs_flat[:, in_int[temp_ind]] = np.roll(dirs_flat[:, in_int[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p_ranges_interface.p3_range) & (randomise <= self.p_ranges_interface.p4_range))[0], dtype=np.uint32)
        dirs_flat[:, in_int[temp_ind]] = np.roll(dirs_flat[:, in_int[temp_ind]], 2, axis=0)
        dirs_flat[:, in_int[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p_ranges_interface.p4_range) & (randomise <= self.p_ranges_interface.p_r_range))[0], dtype=np.uint32)
        dirs_flat[:, in_int[temp_ind]] *= -1

        # IN scale Diffusion
        randomise = np.array(np.random.random_sample(blocked.size), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p_ranges_scale.p1_range)[0], dtype=np.uint32)
        dirs_flat[:, blocked[temp_ind]] = np.roll(dirs_flat[:, blocked[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p1_range) & (randomise <= self.p_ranges_scale.p2_range))[0], dtype=np.uint32)
        dirs_flat[:, blocked[temp_ind]] = np.roll(dirs_flat[:, blocked[temp_ind]], 1, axis=0)
        dirs_flat[:, blocked[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p2_range) & (randomise <= self.p_ranges_scale.p3_range))[0], dtype=np.uint32)
        dirs_flat[:, blocked[temp_ind]] = np.roll(dirs_flat[:, blocked[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p3_range) & (randomise <= self.p_ranges_scale.p4_range))[0], dtype=np.uint32)
        dirs_flat[:, blocked[temp_ind]] = np.roll(dirs_flat[:, blocked[temp_ind]], 2, axis=0)
        dirs_flat[:, blocked[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p4_range) & (randomise <= self.p_ranges_scale.p_r_range))[0], dtype=np.uint32)
        dirs_flat[:, blocked[temp_ind]] *= -1

        cells_flat = np.add(cells_flat, dirs_flat, casting="unsafe")
        # Adjust coordinates for boundary conditions
        ind = np.where(cells_flat[2] < 0)[0]
        # open left bound
        keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
        keep_mask[ind] = False
        cells_flat = cells_flat[:, keep_mask]
        dirs_flat = dirs_flat[:, keep_mask]

        cells_flat[0, np.where(cells_flat[0] <= -1)] = self.cells_per_axis - 1
        cells_flat[0, np.where(cells_flat[0] >= self.cells_per_axis)] = 0
        cells_flat[1, np.where(cells_flat[1] <= -1)] = self.cells_per_axis - 1
        cells_flat[1, np.where(cells_flat[1] >= self.cells_per_axis)] = 0

        ind = np.where(cells_flat[2] >= self.cells_per_axis)[0]
        # open right bound
        keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
        keep_mask[ind] = False
        cells_flat = cells_flat[:, keep_mask]
        dirs_flat = dirs_flat[:, keep_mask]

        self._set_flat_cells_dirs(cells_flat, dirs_flat)
        self.current_count = len(np.where(cells_flat[2] == 0)[0]) if cells_flat.shape[1] > 0 else 0
        self.fill_first_page()

    def fill_first_page(self, time=0):
        """
        Generate new particles on the diffusion surface (z = 0, k = 0).
        Works directly with grid (_diff_A_count, _diff_A_dirs) - no flat arrays.
        """
        if not _DIFFUSION_SHM_AVAILABLE or self._diff_A_count is None:
            return
        
        n = self.cells_per_axis
        max_per_cell = self._diff_max_per_cell
        rng = np.random.default_rng()
        
        # Get current read buffer
        count = self._diff_A_count if self._diff_read_name == self._diff_shm_A.name else self._diff_B_count
        dirs = self._diff_A_dirs if self._diff_read_name == self._diff_shm_A.name else self._diff_B_dirs
        
        # Count particles at z=0 (k=0) - first n² indices
        n2 = n * n
        self.current_count = int(count[:n2].sum())
        adj_cells_pro_page = self.n_per_page - self.current_count
        
        if adj_cells_pro_page > 0:
            # Direction (0, 0, 1) packed: (0+1) + (0+1)*4 + (1+1)*16 = 1 + 4 + 32 = 37
            dir_packed = np.uint8(37)  # Inward direction (positive z)
            
            # Generate random (i, j) positions on z=0 plane
            for _ in range(adj_cells_pro_page):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                idx = i + n * j  # k=0, so idx = i + n*j + n²*0 = i + n*j
                
                # Add particle if cell is not full
                if count[idx] < max_per_cell:
                    count[idx] += 1
                    dirs[idx, count[idx] - 1] = dir_packed

    def _init_diffusion_buffers_oxidant(self):
        """Create shared-memory diffusion grids for inward diffusion."""
        n = self.cells_per_axis
        max_per_cell = self._diff_max_per_cell
        
        # Create element-specific buffers
        # Note: DiffusionParameters are now handled internally by DiffusionEngine
        (self._diff_shm_A, self._diff_shm_B,
         self._diff_A_count, self._diff_A_dirs,
         self._diff_B_count, self._diff_B_dirs) = create_diffusion_buffers(n, max_per_cell)
        self._diff_read_name = self._diff_shm_A.name
        self._diff_write_name = self._diff_shm_B.name
        
        # Store only element-specific probability values
        p1 = self.p1_range
        p_r_extra = self.p_r_range - self.p4_range
        self._diff_step_args = {
            "p1": p1,
            "p2": 2 * p1,
            "p3": 3 * p1,
            "p4": 4 * p1,
            "p_r": 4 * p1 + p_r_extra,
        }

    def _get_current_grid(self):
        """Get current read buffer (count, dirs) from grid."""
        read_count = self._diff_A_count if self._diff_read_name == self._diff_shm_A.name else self._diff_B_count
        read_dirs = self._diff_A_dirs if self._diff_read_name == self._diff_shm_A.name else self._diff_B_dirs
        return read_count, read_dirs

    def get_diffusion_state(self):
        """Return current diffusion state for DiffusionEngine (DiffusibleElement protocol)."""
        args = self._diff_step_args
        return {
            'read_name': self._diff_read_name,
            'write_name': self._diff_write_name,
            'p1': args["p1"],
            'p2': args["p2"],
            'p3': args["p3"],
            'p4': args["p4"],
            'p_r': args["p_r"],
        }
    
    def get_diffusion_config(self):
        """Return diffusion configuration (DiffusibleElement protocol)."""
        p1 = self.p1_range
        p_r_extra = self.p_r_range - self.p4_range
        return {
            'max_per_cell': self._diff_max_per_cell,
            'element_type': 'inward',  # OxidantElem is inward diffusion
            'p1': p1,
            'p_r_extra': p_r_extra,
        }
    
    def swap_diffusion_buffers(self):
        """Swap read/write buffers after diffusion step (DiffusibleElement protocol)."""
        self._diff_read_name, self._diff_write_name = self._diff_write_name, self._diff_read_name

    def get_diffusion_grid_3d(self, copy=False):
        """
        Return the current diffusion read buffer as 3D arrays.
        
        Returns:
            count_3d: (n, n, n) int8 – particle count per cell
            dirs_3d: (n, n, n, max_per_cell) uint8 – packed direction per cell/slot
        If copy=True returns copies; otherwise returns views (same memory as shared buffer).
        """
        read_count, read_dirs = self._get_current_grid()
        n = self.cells_per_axis
        max_per_cell = self._diff_max_per_cell
        count_3d = read_count.reshape(n, n, n)
        dirs_3d = read_dirs.reshape(n, n, n, max_per_cell)
        if copy:
            return count_3d.copy(), dirs_3d.copy()
        return count_3d, dirs_3d

    @staticmethod
    def generate_prob_ranges(probabilities):
        p1_range = probabilities[0]
        p2_range = 2 * p1_range
        p3_range = 3 * p1_range
        p4_range = 4 * p1_range
        p_r_range = p4_range + probabilities[1]
        return PRanges(p1_range, p2_range, p3_range, p4_range, p_r_range)

    def close_and_unlink_shm(self):
        if not self.shms_unlinked:
            # c3d_shared removed - no longer needed
            if _DIFFUSION_SHM_AVAILABLE and self._diff_shm_A is not None:
                self._diff_shm_A.close()
                self._diff_shm_A.unlink()
                self._diff_shm_B.close()
                self._diff_shm_B.unlink()
            self.shms_unlinked = True


class Product:
    def __init__(self, settings):
        self.constitution = settings.CONSTITUTION
        cells_per_axis = Config.N_CELLS_PER_AXIS
        self.shape = (cells_per_axis, cells_per_axis, cells_per_axis + 1)
        self.oxidation_number = settings.OXIDATION_NUMBER
        self.lind_flat_arr = settings.LIND_FLAT_ARRAY

        if self.oxidation_number == 1:
            self.fix_full_cells = self.fix_full_cells_ox_numb_single
            self.transform_c3d = self.transform_c3d_single
        else:
            self.fix_full_cells = self.fix_full_cells_ox_numb_mult
            self.transform_c3d = self.transform_c3d_mult

        temp = np.full(self.shape, 0, dtype=np.ubyte)
        self.c3d_shared = shared_memory.SharedMemory(create=True, size=temp.nbytes)
        self.c3d = np.ndarray(self.shape, dtype=np.ubyte, buffer=self.c3d_shared.buf)
        self.c3d_shm_mdata = SharedMetaData(self.c3d_shared.name, self.shape, np.ubyte)

        temp = np.full((self.shape[0], self.shape[1], self.shape[2] - 1), 0, dtype=bool)
        self.full_c3d_shared = shared_memory.SharedMemory(create=True, size=temp.nbytes)
        self.full_c3d = np.ndarray((self.shape[0], self.shape[1], self.shape[2] - 1), dtype=bool,
                                   buffer=self.full_c3d_shared.buf)

        full_c3d_shared_shape = (self.shape[0], self.shape[1], self.shape[2] - 1)
        self.full_shm_mdata = SharedMetaData(self.full_c3d_shared.name, full_c3d_shared_shape, bool)

        self.shms_unlinked = False

    def fix_full_cells_ox_numb_single(self, new_precip):
        self.full_c3d[new_precip[0], new_precip[1], new_precip[2]] = True

    def fix_full_cells_ox_numb_mult(self, new_precip):
        current_precip = np.array(self.c3d[new_precip[0], new_precip[1], new_precip[2]], dtype=np.ubyte)
        indexes = np.where(current_precip == self.oxidation_number)[0]
        full_precip = new_precip[:, indexes]
        self.full_c3d[full_precip[0], full_precip[1], full_precip[2]] = True

    def transform_c3d_single(self):
        return np.array(np.nonzero(self.c3d), dtype=np.short)

    def transform_c3d_mult(self):
        precipitations = np.array(np.nonzero(self.c3d), dtype=np.short)
        counts = self.c3d[precipitations[0], precipitations[1], precipitations[2]]
        return np.array(np.repeat(precipitations, counts, axis=1), dtype=np.short)

    def close_and_unlink_shm(self):
        if not self.shms_unlinked:
            self.c3d_shared.close()
            self.c3d_shared.unlink()
            self.full_c3d_shared.close()
            self.full_c3d_shared.unlink()
            self.shms_unlinked = True
