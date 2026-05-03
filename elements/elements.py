from utils.numba_functions import *
from configuration import Config
from multiprocessing import shared_memory
from cellular_automata.nes_for_mp import *
import numpy as np


try:
    from diffusion_3d_mp_example import (
        _DIRS_6_PACKED,
        _views_from_segment,  # Helper for buffer creation
    )
    _DIFFUSION_SHM_AVAILABLE = True
except ImportError:
    _DIFFUSION_SHM_AVAILABLE = False
    DiffusionEngine = None
    DiffusibleElement = None
    _DIRS_6_PACKED = None


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
    # NOTE: count stores per-cell occupancy up to max_per_cell (often > 127).
    # int8 would overflow and corrupt both counts and dirs indexing.
    count_dtype = np.uint16
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


def _prob_triplet_to_thresholds(prob_triplet):
    p1 = float(prob_triplet[0])
    p2 = 2.0 * p1
    p3 = 3.0 * p1
    p4 = 4.0 * p1
    p_r = p4 + float(prob_triplet[1])
    return p1, p2, p3, p4, p_r


def _build_product_phase_threshold_map(settings, fallback_thresholds):
    raw_map = getattr(settings, "PRODUCT_PROBABILITIES_BY_PHASE", {}) or {}
    out = {}
    for raw_pid, raw_probs in raw_map.items():
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        out[pid] = _prob_triplet_to_thresholds(raw_probs)
    if len(out) == 0:
        return {}
    for pid in list(out.keys()):
        if out[pid] is None:
            out[pid] = fallback_thresholds
    return out


class ActiveElem:
    element_type = 'outward'
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
        (self.p1_in_product, self.p2_in_product, self.p3_in_product,
         self.p4_in_product, self.p_r_in_product) = _prob_triplet_to_thresholds(
            getattr(settings, "PROBABILITIES_IN_PRODUCT", settings.PROBABILITIES)
        )
        self.product_phase_probabilities = _build_product_phase_threshold_map(
            settings,
            (self.p1_in_product, self.p2_in_product, self.p3_in_product, self.p4_in_product, self.p_r_in_product),
        )
        self.use_product_aware_diffusion = any(
            abs(vals[0] - self.p1_range) > 0.0 or abs(vals[4] - self.p_r_range) > 0.0
            for vals in self.product_phase_probabilities.values()
        )
        self.n_per_page = settings.N_PER_PAGE

        self.p_ranges = PRanges(self.p1_range, self.p2_range, self.p3_range, self.p4_range, self.p_r_range)
        self.p_ranges_scale = PRanges(self.p1_range, self.p2_range, self.p3_range, self.p4_range, self.p_r_range)

        extended_axis = self.cells_per_axis + self.neigh_range
        self.extended_shape = (self.cells_per_axis, self.cells_per_axis, extended_axis)

        self.diffuse = None  # must be defined elsewhere
        self.scale = None  # must be defined elsewhere

        self.current_count = None
        self.shms_unlinked = False
        self.skip_diffusion_this_step = False  # set by CA diffuse_all when STRIDE says skip outward this step

        self.max_per_cell = settings.DIFFUSION_MAX_PER_CELL
        self._diff_shm_A = self._diff_shm_B = None
        self._diff_A_count = self._diff_A_dirs = self._diff_B_count = self._diff_B_dirs = None
        self._diff_read_name = self._diff_write_name = None
        self._diff_step_args = None
        
        self._init_diffusion_buffers()
        self._init_particles_in_grid(settings)

    def _init_diffusion_buffers(self):
        """Create shared-memory diffusion grids (count + dirs) and prepare run."""
        n = self.cells_per_axis
        max_per_cell = self.max_per_cell
        
        # Create element-specific buffers
        # Note: DiffusionParameters are now handled internally by DiffusionEngine
        (self._diff_shm_A, self._diff_shm_B,
         self._diff_A_count, self._diff_A_dirs,
         self._diff_B_count, self._diff_B_dirs) = create_diffusion_buffers(n, max_per_cell)
        self._diff_read_name = self._diff_shm_A.name
        self._diff_write_name = self._diff_shm_B.name
        # Expose count buffer as c3d_shm_mdata for nucleation/precipitation (same shm, first segment)
        self.c3d_shm_mdata = SharedMetaData(
            self._diff_shm_A.name, (n, n, n), np.uint16
        )
        self.cells_shm_mdata = None  # legacy flat cells; not used when USE_NEW_DIFFUSION_ENGINE
        self.dirs_shm_mdata = None

    def get_current_c3d_shm_mdata(self):
        """Return SharedMetaData for the current diffusion *read* buffer (after swap). Use for nucleation so it reads up-to-date grid."""
        n = self.cells_per_axis
        name = self._diff_shm_A.name if self._diff_read_name == self._diff_shm_A.name else self._diff_shm_B.name
        return SharedMetaData(name, (n, n, n), np.uint16)

    def _init_particles_in_grid(self, settings):
        """Initialize particles directly in the 3D grid (no flat arrays). CONC_PRECISION: 'rand' or 'exact'. SPACE_FILL: 'full' or 'half'. Uses Numba JIT for speed."""
        n = self.cells_per_axis
        max_per_cell = self.max_per_cell
        count = self._diff_A_count
        dirs = self._diff_A_dirs
        count.fill(0)
        dirs.fill(0)
        n2 = n * n
        i_lo = n // 2 if getattr(settings, 'SPACE_FILL', 'full').lower() == 'half' else 0
        packed_dirs = _DIRS_6_PACKED if _DIRS_6_PACKED is not None else np.array([20, 22, 17, 25, 5, 37], dtype=np.uint8)
        seed = int(np.random.default_rng().integers(0, 2**31))

        if settings.CONC_PRECISION.lower() == 'rand':
            num_x_slices = n - i_lo  # full grid: n slices; half (x): n - n//2
            total_particles = int(self.n_per_page * num_x_slices)
            init_particles_rand(count, dirs, n, n2, max_per_cell, total_particles, packed_dirs, i_lo, seed)
        elif settings.CONC_PRECISION.lower() == 'exact':
            n_per = min(int(self.n_per_page), n * n)
            init_particles_exact(count, dirs, n, n2, max_per_cell, n_per, i_lo, packed_dirs, seed)
        else:
            raise ValueError(f"Wrong CONC_PRECISION for outward element! (use 'exact' or 'rand')")

    def _get_current_grid(self):
        """Get current read buffer (count, dirs) from grid."""
        read_count = self._diff_A_count if self._diff_read_name == self._diff_shm_A.name else self._diff_B_count
        read_dirs = self._diff_A_dirs if self._diff_read_name == self._diff_shm_A.name else self._diff_B_dirs
        return read_count, read_dirs

    def fill_last_page(self, time=0):
        """
        If current particle count on the last x-plane (x=n-1) is less than n_per_page, add the difference
        by sampling from currently empty slots. Uses current read buffer. Concentration target comes from
        Config ACTIVES.*.CELLS_CONCENTRATION (n_per_page is set from that at init).
        """
        count, dirs = self._get_current_grid()
        n = self.cells_per_axis
        n2 = n * n
        max_per_cell = self.max_per_cell
        count_3d = count.reshape(n, n, n, order="F")
        current_count = int(count_3d[n - 1, :, :].sum())
        if current_count >= self.n_per_page:
            return
        num_to_add = self.n_per_page - current_count
        rng = np.random.default_rng()
        j_coords = rng.integers(0, n, size=num_to_add, dtype=np.intp)
        k_coords = rng.integers(0, n, size=num_to_add, dtype=np.intp)
        packed_dirs = np.array([22], dtype=np.uint8)
        dir_packed = rng.choice(packed_dirs, size=num_to_add)
        not_inserted = fill_last_page_kernel(count, dirs, n, n2, max_per_cell, j_coords, k_coords, dir_packed)
        if not_inserted > 0:
            print(f"Warning: {not_inserted} particles not inserted into last page (outward).")

    def get_diffusion_state(self):
        """Return current diffusion state for DiffusionEngine (DiffusibleElement protocol)."""
        return {
            'read_name': self._diff_read_name,
            'write_name': self._diff_write_name,
            'p1': self.p1_range,
            'p2': self.p2_range,
            'p3': self.p3_range,
            'p4': self.p4_range,
            'p_r': self.p_r_range,
            'use_product_aware_diffusion': self.use_product_aware_diffusion,
            'product_phase_probabilities': self.product_phase_probabilities,
        }
    
    def swap_diffusion_buffers(self):
        """Swap read/write buffers after diffusion step (DiffusibleElement protocol)."""
        self._diff_read_name, self._diff_write_name = self._diff_write_name, self._diff_read_name

    def get_3d_grid(self, copy=False):
        """
        Return the current diffusion read buffer as 3D arrays.
        
        Returns:
            count_3d: (n, n, n) int8 – particle count per cell
            dirs_3d: (n, n, n, max_per_cell) uint8 – packed direction per cell/slot
        If copy=True returns copies; otherwise returns views (same memory as shared buffer).
        """
        read_count, read_dirs = self._get_current_grid()
        n = self.cells_per_axis
        max_per_cell = self.max_per_cell
        count_3d = read_count.reshape(n, n, n, order="F")
        dirs_3d = read_dirs.reshape(n, n, n, max_per_cell)
        if copy:
            return count_3d.copy(), dirs_3d.copy()
        return count_3d, dirs_3d

    def get_cells_coords(self):
        """
        Return particle coordinates for DB/results: shape (3, N) with rows (z, y, x)
        from the current diffusion read buffer. Used by save_results when using the new 3D grid.
        """
        read_count, _ = self._get_current_grid()
        n = self.cells_per_axis
        count_3d = read_count.reshape(n, n, n, order="F")
        idx = np.nonzero(count_3d)
        if len(idx[0]) == 0:
            return np.zeros((3, 0), dtype=np.short)
        counts = count_3d[idx].astype(np.intp)
        i, j, k = idx[0], idx[1], idx[2]
        z = np.repeat(k, counts)
        y = np.repeat(j, counts)
        x = np.repeat(i, counts)
        return np.stack([z, y, x], axis=0).astype(np.short)

    def transform_to_descards(self):
        """No-op for 3D grid representation (legacy flat-array path used transform before diffuse)."""
        pass

    def transform_to_3d(self, _depth=None):
        """No-op for 3D grid representation (legacy path converted flat back to 3D after save)."""
        pass

    def close_and_unlink_shm(self):
        if not self.shms_unlinked:
            self._diff_shm_A.close()
            self._diff_shm_A.unlink()
            self._diff_shm_B.close()
            self._diff_shm_B.unlink()
            self.shms_unlinked = True


class OxidantElem:
    element_type = 'inward'
    def __init__(self, settings, utils):
        self.elem_name = settings.ELEMENT
        self.cells_per_axis = Config.N_CELLS_PER_AXIS
        self.p1_range = settings.PROBABILITIES[0]
        self.p2_range = 2 * self.p1_range
        self.p3_range = 3 * self.p1_range
        self.p4_range = 4 * self.p1_range
        self.p_r_range = self.p4_range + settings.PROBABILITIES[1]
        (self.p1_in_product, self.p2_in_product, self.p3_in_product,
         self.p4_in_product, self.p_r_in_product) = _prob_triplet_to_thresholds(
            getattr(settings, "PROBABILITIES_IN_PRODUCT", settings.PROBABILITIES)
        )
        self.product_phase_probabilities = _build_product_phase_threshold_map(
            settings,
            (self.p1_in_product, self.p2_in_product, self.p3_in_product, self.p4_in_product, self.p_r_in_product),
        )
        self.use_product_aware_diffusion = any(
            abs(vals[0] - self.p1_range) > 0.0 or abs(vals[4] - self.p_r_range) > 0.0
            for vals in self.product_phase_probabilities.values()
        )
        self.p0_2d = settings.PROBABILITIES_2D
        self.n_per_page = settings.N_PER_PAGE
        self.neigh_range = Config.NEIGH_RANGE
        self.current_count = 0
        self.furthest_index = None

        self.p_ranges_scale = PRanges(self.p1_range, self.p2_range, self.p3_range, self.p4_range, self.p_r_range)
        self.p_ranges_interface = PRanges(self.p1_range, self.p2_range, self.p3_range, self.p4_range, self.p_r_range)

        self.extended_axis = self.cells_per_axis + self.neigh_range
        self.extended_shape = (self.cells_per_axis, self.cells_per_axis, self.extended_axis)

        self.shms_unlinked = False

        self.scale = None
        self.diffuse = None
        self.n_boost_steps = Config.N_BOOST_STEPS

        self.utils = utils
        self.microstructure = None

        self.max_per_cell = settings.DIFFUSION_MAX_PER_CELL
        self._diff_shm_A = self._diff_shm_B = None
        self._diff_A_count = self._diff_A_dirs = self._diff_B_count = self._diff_B_dirs = None
        self._diff_read_name = self._diff_write_name = None
        self._diff_step_args = None
        
        self._init_diffusion_buffers_oxidant()
        # Initialize with empty grid (fill_first_page will add particles)
        self.current_count = 0
        self.from_product_counts = 0
        self.adjusted_cells = settings.N_PER_PAGE
        self.k_const = settings.K_CONST
        self.numbs_to_add = []
        self.fill_first_page()

        # self.microstructure = voronoi.VoronoiMicrostructure(self.cells_per_axis)
        # self.microstructure.generate_voronoi_3d(50, seeds="own")
        # self.microstructure.show_microstructure(self.cells_per_axis)
        # self.cross_shifts = np.array([[1, 0, 0], [0, 1, 0],
        #                               [-1, 0, 0], [0, -1, 0],
        #                               [0, 0, -1]], dtype=np.byte)

    def fill_first_page(self, time=0):
        """
        If current particle count on x=0 plane is less than n_per_page, add the difference
        by sampling from currently empty slots. If already >= n_per_page, do nothing.
        Uses the current read buffer (same as diffusion step) so we add to the buffer that is actually read.
        """
        count = self._diff_A_count if self._diff_read_name == self._diff_shm_A.name else self._diff_B_count
        dirs = self._diff_A_dirs if self._diff_read_name == self._diff_shm_A.name else self._diff_B_dirs

        n = self.cells_per_axis
        n2 = n * n
        max_per_cell = self.max_per_cell
        count_3d = count.reshape(n, n, n, order="F")
        # Same layout as diffusion (i,j,k)=(x,y,z): x=0 plane is count_3d[0, :, :], linear index n*j + n2*k
        current_count = int(count_3d[0, :, :].sum())

        if current_count >= self.adjusted_cells:
            return

        num_to_add = self.adjusted_cells - current_count
        self.numbs_to_add.append(num_to_add)
        rng = np.random.default_rng()
        j_coords = rng.integers(0, n, size=num_to_add, dtype=np.intp)
        k_coords = rng.integers(0, n, size=num_to_add, dtype=np.intp)
        # packed_dirs = _DIRS_6_PACKED if _DIRS_6_PACKED is not None else np.array([20, 22, 17, 25, 5, 37], dtype=np.uint8)
        packed_dirs = np.array([22], dtype=np.uint8)
        dir_packed = rng.choice(packed_dirs, size=num_to_add)
        not_inserted = fill_first_page_kernel(count, dirs, n, n2, max_per_cell, j_coords, k_coords, dir_packed)
        if not_inserted > 0:
            print(f"Warning: {not_inserted} particles not inserted into first page.")


    def _init_diffusion_buffers_oxidant(self):
        """Create shared-memory diffusion grids for inward diffusion."""
        n = self.cells_per_axis
        max_per_cell = self.max_per_cell
        
        # Create element-specific buffers
        # Note: DiffusionParameters are now handled internally by DiffusionEngine
        (self._diff_shm_A, self._diff_shm_B,
         self._diff_A_count, self._diff_A_dirs,
         self._diff_B_count, self._diff_B_dirs) = create_diffusion_buffers(n, max_per_cell)
        self._diff_read_name = self._diff_shm_A.name
        self._diff_write_name = self._diff_shm_B.name
        # Expose count buffer as c3d_shm_mdata for engine/case_mp (same shm, first segment)
        self.c3d_shm_mdata = SharedMetaData(self._diff_shm_A.name, (n, n, n), np.uint16)

    def get_current_c3d_shm_mdata(self):
        """Return SharedMetaData for the current diffusion *read* buffer (after swap). Use for nucleation so it reads up-to-date grid."""
        n = self.cells_per_axis
        name = self._diff_shm_A.name if self._diff_read_name == self._diff_shm_A.name else self._diff_shm_B.name
        return SharedMetaData(name, (n, n, n), np.uint16)

    def _get_current_grid(self):
        """Get current read buffer (count, dirs) from grid."""
        read_count = self._diff_A_count if self._diff_read_name == self._diff_shm_A.name else self._diff_B_count
        read_dirs = self._diff_A_dirs if self._diff_read_name == self._diff_shm_A.name else self._diff_B_dirs
        return read_count, read_dirs

    def get_diffusion_state(self):
        """Return current diffusion state for DiffusionEngine (DiffusibleElement protocol)."""
        return {
            'read_name': self._diff_read_name,
            'write_name': self._diff_write_name,
            'p1': self.p1_range,
            'p2': self.p2_range,
            'p3': self.p3_range,
            'p4': self.p4_range,
            'p_r': self.p_r_range,
            'use_product_aware_diffusion': self.use_product_aware_diffusion,
            'product_phase_probabilities': self.product_phase_probabilities,
        }
    
    def swap_diffusion_buffers(self):
        """Swap read/write buffers after diffusion step (DiffusibleElement protocol)."""
        self._diff_read_name, self._diff_write_name = self._diff_write_name, self._diff_read_name

    def get_3d_grid(self, copy=False):
        """
        Return the current diffusion read buffer as 3D arrays.
        
        Returns:
            count_3d: (n, n, n) int8 – particle count per cell
            dirs_3d: (n, n, n, max_per_cell) uint8 – packed direction per cell/slot
        If copy=True returns copies; otherwise returns views (same memory as shared buffer).
        """
        read_count, read_dirs = self._get_current_grid()
        n = self.cells_per_axis
        max_per_cell = self.max_per_cell
        count_3d = read_count.reshape(n, n, n, order="F")
        dirs_3d = read_dirs.reshape(n, n, n, max_per_cell)
        if copy:
            return count_3d.copy(), dirs_3d.copy()
        return count_3d, dirs_3d

    def get_cells_coords(self):
        """
        Return particle coordinates for DB/results: shape (3, N) with rows (z, y, x)
        from the current diffusion read buffer. Used by save_results when using the new 3D grid.
        """
        read_count, _ = self._get_current_grid()
        n = self.cells_per_axis
        count_3d = read_count.reshape(n, n, n, order="F")
        idx = np.nonzero(count_3d)
        if len(idx[0]) == 0:
            return np.zeros((3, 0), dtype=np.short)
        counts = count_3d[idx].astype(np.intp)
        i, j, k = idx[0], idx[1], idx[2]
        z = np.repeat(k, counts)
        y = np.repeat(j, counts)
        x = np.repeat(i, counts)
        return np.stack([z, y, x], axis=0).astype(np.short)

    @property
    def cells(self):
        """Particle coords (3, N) for DB save; same format as get_cells_coords()."""
        return self.get_cells_coords()

    def transform_to_descards(self):
        """No-op for 3D grid representation (legacy flat-array path)."""
        pass

    def transform_to_3d(self):
        """No-op for 3D grid representation (legacy path)."""
        pass

    def close_and_unlink_shm(self):
        if not self.shms_unlinked:
            self._diff_shm_A.close()
            self._diff_shm_A.unlink()
            self._diff_shm_B.close()
            self._diff_shm_B.unlink()
            self.shms_unlinked = True
