from utils.numba_functions import *
from configuration import Config
from multiprocessing import shared_memory
from cellular_automata.nes_for_mp import *
import sys
import random
import numpy as np

# Optional: 3D Chopard–Droz diffusion module (shared-memory, multiprocessing)
# 
# Architecture: Elements implement DiffusibleElement protocol and expose their state.
# DiffusionEngine applies diffusion externally (like shuffling a Rubik's cube).
# 
# New usage (recommended):
#   from diffusion_3d_mp_example import DiffusionEngine
#   engine = DiffusionEngine(pool, rng)
#   engine.diffuse(active_elem)  # Apply diffusion step
#
# Legacy usage (still supported):
#   active_elem.diffuse_step_mp(pool, rng)  # Creates engine internally
#
try:
    from diffusion_3d_mp_example import (
        create_diffusion_buffers,
        prepare_diffusion_run,
        flat_to_grid_sync,
        grid_to_flat_sync,
        DiffusionEngine,
        DiffusibleElement,
        _parse_boundary,
        _idx,
        _DIRS_6_PACKED,
    )
    _DIFFUSION_SHM_AVAILABLE = True
except ImportError:
    _DIFFUSION_SHM_AVAILABLE = False
    DiffusionEngine = None
    DiffusibleElement = None
    # Fallback if module not available
    def _idx(i, j, k, n):
        return int(i) + int(n) * (int(j) + int(n) * int(k))
    _DIRS_6_PACKED = None


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
        self._diff_n_workers = getattr(Config, 'OUTWARD_DIFFUSION_WORKERS', 7)
        self._diff_boundary_x = getattr(Config, 'DIFFUSION_BOUNDARY_X', 'periodic')
        self._diff_shm_A = self._diff_shm_B = None
        self._diff_A_count = self._diff_A_dirs = self._diff_B_count = self._diff_B_dirs = None
        self._diff_read_name = self._diff_write_name = None
        self._diff_step_args = None
        
        # Temporary flat arrays for conversion (not persistent storage)
        self._temp_cells = None
        self._temp_dirs = None
        
        if _DIFFUSION_SHM_AVAILABLE:
            self._init_diffusion_buffers()
            # Initialize grid with particles based on CONC_PRECISION and SPACE_FILL
            self._init_particles_in_grid(settings)

    def _init_diffusion_buffers(self):
        """Create shared-memory diffusion grids (count + dirs) and prepare run."""
        n = self.cells_per_axis
        max_per_cell = self._diff_max_per_cell
        (self._diff_shm_A, self._diff_shm_B,
         self._diff_A_count, self._diff_A_dirs,
         self._diff_B_count, self._diff_B_dirs,
         count_bytes, dirs_bytes) = create_diffusion_buffers(n, max_per_cell)
        self._diff_read_name = self._diff_shm_A.name
        self._diff_write_name = self._diff_shm_B.name
        p1 = self.p1_range
        p_r_extra = self.p_r_range - self.p4_range
        prep = prepare_diffusion_run(
            n, self._diff_n_workers, max_per_cell,
            self._diff_boundary_x, p1, p_r_extra
        )
        self._diff_step_args = {
            "n": n,
            "max_per_cell": max_per_cell,
            "count_bytes": prep["count_bytes"],
            "dirs_bytes": prep["dirs_bytes"],
            "subblock_arg_templates": prep["subblock_arg_templates"],
            "gap_groups": prep["gap_groups"],
            "kernel_idx": prep["kernel_idx"],
            "p1": prep["p1_val"],
            "p2": prep["p2_val"],
            "p3": prep["p3_val"],
            "p4": prep["p4_val"],
            "p_r": prep["p_r_val"],
        }

    def _init_particles_in_grid(self, settings):
        """Initialize particles in the 3D grid based on CONC_PRECISION and SPACE_FILL."""
        if not _DIFFUSION_SHM_AVAILABLE or self._diff_A_count is None:
            return
        n = self.cells_per_axis
        max_per_cell = self._diff_max_per_cell
        rng = np.random.default_rng()
        
        # Generate initial flat arrays
        if settings.CONC_PRECISION.lower() == 'rand':
            total_particles = int(self.n_per_page * self.cells_per_axis)
            cells_flat = np.random.randint(0, n, size=(3, total_particles), dtype=np.int16)
        elif settings.CONC_PRECISION.lower() == 'exact':
            cells_flat = np.array([[], [], []], dtype=np.int16)
            for plane_xind in range(self.cells_per_axis):
                new_cells = np.array(random.sample(range(self.cells_per_axis**2), int(self.n_per_page)))
                new_cells = np.array(np.unravel_index(new_cells, (self.cells_per_axis, self.cells_per_axis)))
                new_cells = np.vstack((new_cells, np.full(len(new_cells[0]), plane_xind)))
                cells_flat = np.concatenate((cells_flat, new_cells), axis=1)
        else:
            raise ValueError(f"Wrong CONC_PRECISION value for outward element! (possible 'exact' or 'rand')!")
        
        # Apply SPACE_FILL filter
        if settings.SPACE_FILL == 'half':
            ind_to_keep = np.where(cells_flat[2] >= int(self.cells_per_axis / 2))[0]
            cells_flat = cells_flat[:, ind_to_keep]
        
        # Generate random directions
        if _DIRS_6_PACKED is not None:
            dirs_flat = np.array([_DIRS_6_PACKED[rng.integers(0, 6)] for _ in range(cells_flat.shape[1])], dtype=np.int8)
            # Unpack to (3, N) format
            dirs_flat_3d = np.zeros((3, cells_flat.shape[1]), dtype=np.int8)
            for i in range(cells_flat.shape[1]):
                b = dirs_flat[i]
                dirs_flat_3d[0, i] = (b & 3) - 1
                dirs_flat_3d[1, i] = ((b >> 2) & 3) - 1
                dirs_flat_3d[2, i] = ((b >> 4) & 3) - 1
            dirs_flat = dirs_flat_3d
        else:
            dirs_flat = np.random.randint(-1, 2, size=(3, cells_flat.shape[1]), dtype=np.int8)
        
        # Copy to grid
        flat_to_grid_sync(self._diff_A_count, self._diff_A_dirs, cells_flat, dirs_flat, n, max_per_cell)

    def _get_current_grid(self):
        """Get current read buffer (count, dirs) from grid."""
        if not _DIFFUSION_SHM_AVAILABLE or self._diff_step_args is None:
            return None, None
        read_count = self._diff_A_count if self._diff_read_name == self._diff_shm_A.name else self._diff_B_count
        read_dirs = self._diff_A_dirs if self._diff_read_name == self._diff_shm_A.name else self._diff_B_dirs
        return read_count, read_dirs

    def _get_flat_cells_dirs(self):
        """Convert grid to flat arrays (for methods that need flat representation)."""
        if not _DIFFUSION_SHM_AVAILABLE or self._diff_step_args is None:
            return np.zeros((3, 0), dtype=np.int16), np.zeros((3, 0), dtype=np.int8)
        read_count, read_dirs = self._get_current_grid()
        n = self._diff_step_args["n"]
        max_per_cell = self._diff_step_args["max_per_cell"]
        cells_flat, dirs_flat = grid_to_flat_sync(read_count, read_dirs, n, max_per_cell)
        return cells_flat, dirs_flat

    def _set_flat_cells_dirs(self, cells_flat, dirs_flat):
        """Convert flat arrays to grid (write to current write buffer)."""
        if not _DIFFUSION_SHM_AVAILABLE or self._diff_step_args is None:
            return
        write_count = self._diff_B_count if self._diff_read_name == self._diff_shm_A.name else self._diff_A_count
        write_dirs = self._diff_B_dirs if self._diff_read_name == self._diff_shm_A.name else self._diff_A_dirs
        n = self._diff_step_args["n"]
        max_per_cell = self._diff_step_args["max_per_cell"]
        flat_to_grid_sync(write_count, write_dirs, cells_flat, dirs_flat, n, max_per_cell)
        # Swap buffers so the new state becomes read
        self._diff_read_name, self._diff_write_name = self._diff_write_name, self._diff_read_name

    def get_diffusion_state(self):
        """Return current diffusion state for DiffusionEngine (DiffusibleElement protocol)."""
        if not _DIFFUSION_SHM_AVAILABLE or self._diff_step_args is None:
            raise RuntimeError("Diffusion module not available or buffers not initialized")
        args = self._diff_step_args
        return {
            'read_name': self._diff_read_name,
            'write_name': self._diff_write_name,
            'n': args["n"],
            'max_per_cell': args["max_per_cell"],
            'count_bytes': args["count_bytes"],
            'dirs_bytes': args["dirs_bytes"],
            'subblock_arg_templates': args["subblock_arg_templates"],
            'gap_groups': args["gap_groups"],
            'kernel_idx': args["kernel_idx"],
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
            'n_workers': self._diff_n_workers,
            'boundary_x': self._diff_boundary_x,
            'p1': p1,
            'p_r_extra': p_r_extra,
        }
    
    def swap_diffusion_buffers(self):
        """Swap read/write buffers after diffusion step (DiffusibleElement protocol)."""
        self._diff_read_name, self._diff_write_name = self._diff_write_name, self._diff_read_name
    
    def diffuse_step_mp(self, pool, rng):
        """
        Legacy method: One Chopard–Droz diffusion step using multiprocessing.
        DEPRECATED: Use DiffusionEngine.diffuse(element) instead.
        """
        if not _DIFFUSION_SHM_AVAILABLE:
            raise RuntimeError("Diffusion module not available")
        engine = DiffusionEngine(pool, rng)
        engine.diffuse(self)

    def diffuse_bulk(self):
        """Chopard-Droz diffusion through bulk (legacy method, converts grid<->flat)."""
        cells_flat, dirs_flat = self._get_flat_cells_dirs()
        if cells_flat.shape[1] == 0:
            return
        
        # Mixing particles according to Chopard and Droz
        randomise = np.array(np.random.random_sample(cells_flat.shape[1]), dtype=np.single)
        # deflection 1
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        dirs_flat[:, temp_ind] = np.roll(dirs_flat[:, temp_ind], 1, axis=0)
        # deflection 2
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        dirs_flat[:, temp_ind] = np.roll(dirs_flat[:, temp_ind], 1, axis=0)
        dirs_flat[:, temp_ind] *= -1
        # deflection 3
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        dirs_flat[:, temp_ind] = np.roll(dirs_flat[:, temp_ind], 2, axis=0)
        # deflection 4
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        dirs_flat[:, temp_ind] = np.roll(dirs_flat[:, temp_ind], 2, axis=0)
        dirs_flat[:, temp_ind] *= -1
        # reflection
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        dirs_flat[:, temp_ind] *= -1

        cells_flat = np.add(cells_flat, dirs_flat, casting="unsafe")

        # Adjust coordinates for boundary conditions
        ind = np.where(cells_flat[2] < 0)[0]
        # closed left bound (reflection)
        cells_flat[2, ind] = 1
        dirs_flat[2, ind] = 1

        cells_flat[0, np.where(cells_flat[0] == -1)] = self.cells_per_axis - 1
        cells_flat[0, np.where(cells_flat[0] == self.cells_per_axis)] = 0
        cells_flat[1, np.where(cells_flat[1] == -1)] = self.cells_per_axis - 1
        cells_flat[1, np.where(cells_flat[1] == self.cells_per_axis)] = 0

        ind = np.where(cells_flat[2] == self.cells_per_axis)[0]
        # open right bound
        keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
        keep_mask[ind] = False
        cells_flat = cells_flat[:, keep_mask]
        dirs_flat = dirs_flat[:, keep_mask]
        
        self._set_flat_cells_dirs(cells_flat, dirs_flat)
        self.fill_first_page()

    def diffuse_with_scale(self):
        """
        Outward diffusion through bulk + scale (legacy method, converts grid<->flat).
        """
        cells_flat, dirs_flat = self._get_flat_cells_dirs()
        if cells_flat.shape[1] == 0:
            return
        
        # Diffusion through the scale. If the current particle is inside the product particle it will be reflected
        out_scale = check_in_scale(self.scale.full_c3d, cells_flat, dirs_flat)

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
        # closed left bound (reflection)
        cells_flat[2, ind] = 1
        dirs_flat[2, ind] = 1

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
        self.fill_first_page()

    def fill_first_page(self):
        """Generate new particles on the diffusion surface (z = cells_per_axis - 1)."""
        cells_flat, dirs_flat = self._get_flat_cells_dirs()
        self.current_count = len(np.where(cells_flat[2] == self.cells_per_axis - 1)[0]) if cells_flat.shape[1] > 0 else 0
        cells_numb_diff = self.n_per_page - self.current_count
        if cells_numb_diff > 0:
            new_out_page = np.random.randint(self.cells_per_axis, size=(2, cells_numb_diff), dtype=np.int16)
            new_out_page = np.concatenate((new_out_page, np.full((1, cells_numb_diff),
                                                                 self.cells_per_axis - 1, dtype=np.int16)))
            new_dirs = np.zeros((3, cells_numb_diff), dtype=np.int8)
            new_dirs[2, :] = -1
            cells_flat = np.concatenate((cells_flat, new_out_page), axis=1)
            dirs_flat = np.concatenate((dirs_flat, new_dirs), axis=1)
            self._set_flat_cells_dirs(cells_flat, dirs_flat)

    def dell_cells_from_diff_arrays(self, ind_to_del):
        """Delete particles by indices."""
        cells_flat, dirs_flat = self._get_flat_cells_dirs()
        if cells_flat.shape[1] == 0:
            return
        keep_mask = np.ones(cells_flat.shape[1], dtype=bool)
        keep_mask[ind_to_del] = False
        cells_flat = cells_flat[:, keep_mask]
        dirs_flat = dirs_flat[:, keep_mask]
        self._set_flat_cells_dirs(cells_flat, dirs_flat)

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
        self._diff_n_workers = getattr(Config, 'INWARD_DIFFUSION_WORKERS', 3)
        self._diff_boundary_x = getattr(Config, 'DIFFUSION_BOUNDARY_X', 'periodic')
        self._diff_shm_A = self._diff_shm_B = None
        self._diff_A_count = self._diff_A_dirs = self._diff_B_count = self._diff_B_dirs = None
        self._diff_read_name = self._diff_write_name = None
        self._diff_step_args = None
        
        if _DIFFUSION_SHM_AVAILABLE:
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
        Inward diffusion through bulk.
        """
        # # Diffusion along grain boundaries
        # # ______________________________________________________________________________________________________________
        # # exists = self.microstructure.grain_boundaries[self.cells[0], self.cells[1], self.cells[2]]
        # # # print(exists)
        # # temp_ind = np.array(np.where(exists)[0], dtype=np.uint32)
        # # print(temp_ind)
        #
        # exists = self.microstructure.grain_boundaries[self.cells[0], self.cells[1], self.cells[2]]
        # # # print(exists)
        # temp_ind = np.array(np.where(exists)[0], dtype=np.uint32)
        #
        # randomise = np.array(np.random.random_sample(len(temp_ind)), dtype=np.single)
        # d_temp_ind = np.array(np.where(randomise <= self.p0_2d)[0], dtype=np.uint32)
        # temp_ind = temp_ind[d_temp_ind]
        #
        # # print(temp_ind)
        # #
        # in_gb = np.array(self.cells[:, temp_ind], dtype=np.short)
        # # print(in_gb)
        # #
        # shift_vector = np.array(self.microstructure.jump_directions[in_gb[0], in_gb[1], in_gb[2]],
        #                         dtype=np.short).transpose()
        # # print(shift_vector)
        #
        # # print(self.cells)
        # # cross_shifts = np.array(np.random.choice([0, 1, 2, 3], len(shift_vector[0])), dtype=np.ubyte)
        # # cross_shifts = np.array(self.cross_shifts[cross_shifts], dtype=np.byte).transpose()
        #
        # # shift_vector += cross_shifts
        #
        # self.cells[:, temp_ind] += shift_vector
        # # print(self.cells)
        # ______________________________________________________________________________________________________________
        cells_flat, dirs_flat = self._get_flat_cells_dirs()
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
        Inward diffusion through bulk and along grain boundaries (legacy method, converts grid<->flat).
        """
        cells_flat, dirs_flat = self._get_flat_cells_dirs()
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
        Inward diffusion through bulk + scale (legacy method, converts grid<->flat).
        """
        cells_flat, dirs_flat = self._get_flat_cells_dirs()
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
        Inward diffusion through bulk + scale with P (legacy method, converts grid<->flat).
        """
        cells_flat, dirs_flat = self._get_flat_cells_dirs()
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
        Inward diffusion along the phase interfaces (legacy method, converts grid<->flat).
        If the current particle has at least one product particle in its flat neighbourhood and no product ahead
        (in its ballistic direction) it will be boosted forwardly in n_boost_steps.
        """
        cells_flat, dirs_flat = self._get_flat_cells_dirs()
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
        Inward diffusion along the phase interfaces (legacy method, converts grid<->flat).
        If the current particle has at least one product particle in its flat neighbourhood and no product ahead
        (in its ballistic direction) it will be boosted forwardly with higher P.
        """
        cells_flat, dirs_flat = self._get_flat_cells_dirs()
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
        """Generate new particles on the diffusion surface (z = 0)."""
        cells_flat, dirs_flat = self._get_flat_cells_dirs()
        self.current_count = len(np.where(cells_flat[2] == 0)[0]) if cells_flat.shape[1] > 0 else 0
        adj_cells_pro_page = self.n_per_page - self.current_count
        if adj_cells_pro_page > 0:
            new_in_page = np.random.randint(self.cells_per_axis, size=(2, adj_cells_pro_page), dtype=np.int16)
            new_in_page = np.concatenate((new_in_page, np.zeros((1, adj_cells_pro_page), dtype=np.int16)))
            new_dirs = np.zeros((3, adj_cells_pro_page), dtype=np.int8)
            new_dirs[2, :] = 1
            cells_flat = np.concatenate((cells_flat, new_in_page), axis=1)
            dirs_flat = np.concatenate((dirs_flat, new_dirs), axis=1)
            self._set_flat_cells_dirs(cells_flat, dirs_flat)

    def _init_diffusion_buffers_oxidant(self):
        """Create shared-memory diffusion grids for inward diffusion."""
        n = self.cells_per_axis
        max_per_cell = self._diff_max_per_cell
        (self._diff_shm_A, self._diff_shm_B,
         self._diff_A_count, self._diff_A_dirs,
         self._diff_B_count, self._diff_B_dirs,
         count_bytes, dirs_bytes) = create_diffusion_buffers(n, max_per_cell)
        self._diff_read_name = self._diff_shm_A.name
        self._diff_write_name = self._diff_shm_B.name
        p1 = self.p1_range
        p_r_extra = self.p_r_range - self.p4_range
        prep = prepare_diffusion_run(
            n, self._diff_n_workers, max_per_cell,
            self._diff_boundary_x, p1, p_r_extra
        )
        self._diff_step_args = {
            "n": n,
            "max_per_cell": max_per_cell,
            "count_bytes": prep["count_bytes"],
            "dirs_bytes": prep["dirs_bytes"],
            "subblock_arg_templates": prep["subblock_arg_templates"],
            "gap_groups": prep["gap_groups"],
            "kernel_idx": prep["kernel_idx"],
            "p1": prep["p1_val"],
            "p2": prep["p2_val"],
            "p3": prep["p3_val"],
            "p4": prep["p4_val"],
            "p_r": prep["p_r_val"],
        }

    def _get_current_grid(self):
        """Get current read buffer (count, dirs) from grid."""
        if not _DIFFUSION_SHM_AVAILABLE or self._diff_step_args is None:
            return None, None
        read_count = self._diff_A_count if self._diff_read_name == self._diff_shm_A.name else self._diff_B_count
        read_dirs = self._diff_A_dirs if self._diff_read_name == self._diff_shm_A.name else self._diff_B_dirs
        return read_count, read_dirs

    def _get_flat_cells_dirs(self):
        """Convert grid to flat arrays (for methods that need flat representation)."""
        if not _DIFFUSION_SHM_AVAILABLE or self._diff_step_args is None:
            return np.zeros((3, 0), dtype=np.int16), np.zeros((3, 0), dtype=np.int8)
        read_count, read_dirs = self._get_current_grid()
        n = self._diff_step_args["n"]
        max_per_cell = self._diff_step_args["max_per_cell"]
        cells_flat, dirs_flat = grid_to_flat_sync(read_count, read_dirs, n, max_per_cell)
        return cells_flat, dirs_flat

    def _set_flat_cells_dirs(self, cells_flat, dirs_flat):
        """Convert flat arrays to grid (write to current write buffer)."""
        if not _DIFFUSION_SHM_AVAILABLE or self._diff_step_args is None:
            return
        write_count = self._diff_B_count if self._diff_read_name == self._diff_shm_A.name else self._diff_A_count
        write_dirs = self._diff_B_dirs if self._diff_read_name == self._diff_shm_A.name else self._diff_A_dirs
        n = self._diff_step_args["n"]
        max_per_cell = self._diff_step_args["max_per_cell"]
        flat_to_grid_sync(write_count, write_dirs, cells_flat, dirs_flat, n, max_per_cell)
        # Swap buffers so the new state becomes read
        self._diff_read_name, self._diff_write_name = self._diff_write_name, self._diff_read_name

    def get_diffusion_state(self):
        """Return current diffusion state for DiffusionEngine (DiffusibleElement protocol)."""
        if not _DIFFUSION_SHM_AVAILABLE or self._diff_step_args is None:
            raise RuntimeError("Diffusion module not available or buffers not initialized")
        args = self._diff_step_args
        return {
            'read_name': self._diff_read_name,
            'write_name': self._diff_write_name,
            'n': args["n"],
            'max_per_cell': args["max_per_cell"],
            'count_bytes': args["count_bytes"],
            'dirs_bytes': args["dirs_bytes"],
            'subblock_arg_templates': args["subblock_arg_templates"],
            'gap_groups': args["gap_groups"],
            'kernel_idx': args["kernel_idx"],
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
            'n_workers': self._diff_n_workers,
            'boundary_x': self._diff_boundary_x,
            'p1': p1,
            'p_r_extra': p_r_extra,
        }
    
    def swap_diffusion_buffers(self):
        """Swap read/write buffers after diffusion step (DiffusibleElement protocol)."""
        self._diff_read_name, self._diff_write_name = self._diff_write_name, self._diff_read_name

    def calc_furthest_index(self):
        """Get the maximum z-index of particles."""
        cells_flat, _ = self._get_flat_cells_dirs()
        return np.amax(cells_flat[2]) if cells_flat.shape[1] > 0 else -1

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
