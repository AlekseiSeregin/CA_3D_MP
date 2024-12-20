from utils.numba_functions import *
from configuration import Config
from multiprocessing import shared_memory
from cellular_automata.nes_for_mp import *
import sys
import random


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

        self.i_descards = None
        self.i_ind = None

        temp = np.full(self.extended_shape, 0, dtype=np.ubyte)
        self.c3d_shared = shared_memory.SharedMemory(create=True, size=temp.nbytes)
        self.c3d = np.ndarray(self.extended_shape, dtype=np.ubyte, buffer=self.c3d_shared.buf)

        self.c3d_shm_mdata = SharedMetaData(self.c3d_shared.name, self.extended_shape, np.ubyte)
        self.in_3D_flag = False

        self.buffer_reserve = Config.BUFF_SIZE_CONST_ELEM
        self.last_in_diff_arr = int(self.n_per_page * self.cells_per_axis)
        self.diff_arr_buf_size = int(self.last_in_diff_arr * self.buffer_reserve)
        self.diff_arr_shape = (3, self.diff_arr_buf_size)

        # rand/approx concentration space fill
        # ____________________________________________
        if settings.CONC_PRECISION.lower() == 'rand':
            self.last_in_diff_arr = int(self.n_per_page * self.cells_per_axis)
            self.diff_arr_buf_size = int(self.last_in_diff_arr * self.buffer_reserve)
            self.diff_arr_shape = (3, self.diff_arr_buf_size)
            cells = np.random.randint(self.cells_per_axis, size=self.diff_arr_shape, dtype=np.short)
        # ____________________________________________

        # exact concentration space fill
        # ___________________________________________
        elif settings.CONC_PRECISION.lower() == 'exact':
            ex_cells = np.array([[], [], []], dtype=np.short)
            for plane_xind in range(self.cells_per_axis):
                new_cells = np.array(random.sample(range(self.cells_per_axis**2), int(self.n_per_page)))
                new_cells = np.array(np.unravel_index(new_cells, (self.cells_per_axis, self.cells_per_axis)))
                new_cells = np.vstack((new_cells, np.full(len(new_cells[0]), plane_xind)))
                ex_cells = np.concatenate((ex_cells, new_cells), 1)

            self.last_in_diff_arr = len(ex_cells[0])
            self.diff_arr_buf_size = int(self.last_in_diff_arr * self.buffer_reserve)
            cells = np.random.randint(self.cells_per_axis, size=self.diff_arr_shape, dtype=np.short)
            cells[:, :self.last_in_diff_arr] = ex_cells
        # ____________________________________________
        else:
            print("______________________________________________________________")
            print("Wrong CONC_PRECISION value for outward element! (possible 'exact' or 'rand')!")
            print("______________________________________________________________")
            sys.exit()

        self.cells_shm = shared_memory.SharedMemory(create=True, size=cells.nbytes)
        self.cells = np.ndarray(self.diff_arr_shape, dtype=np.short, buffer=self.cells_shm.buf)
        self.cells_shm_mdata = SharedMetaData(self.cells_shm.name, self.diff_arr_shape, np.short)
        np.copyto(self.cells, cells)

        # delete first and second page
        # ____________________________________________
        # f_and_s = np.where((self.cells[2] == 0) | (self.cells[2] == 1))[0]
        # self.cells = np.delete(self.cells, f_and_s, axis=1)
        # ____________________________________________

        # free two first pages (to avoid high concentrations there)
        # ____________________________________________
        # ind = np.where((self.cells[2] == 0) | (self.cells[2] == 1))[0]
        # self.cells = np.delete(self.cells, ind, 1)
        # ____________________________________________
        # free first page (to avoid high concentrations there)
        # ____________________________________________
        # ind = np.where(self.cells[2] == 0)[0]
        # self.cells = np.delete(self.cells, ind, 1)
        # ____________________________________________

        # half space fill
        # ____________________________________________
        if settings.SPACE_FILL == 'half':
            ind_to_del = np.where(self.cells[2, :self.last_in_diff_arr] < int(self.cells_per_axis / 2))[0]
            to_move = np.delete(self.cells[:, :self.last_in_diff_arr], ind_to_del, axis=1)
            self.cells[:, :to_move.shape[1]] = to_move
            self.last_in_diff_arr = to_move.shape[1]
        # ____________________________________________

        dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(self.cells[0]))
        dirs = np.array(np.unravel_index(dirs, (3, 3, 3)), dtype=np.byte)
        dirs -= 1

        self.dirs_shm = shared_memory.SharedMemory(create=True, size=dirs.nbytes)
        self.dirs = np.ndarray(self.diff_arr_shape, dtype=np.byte, buffer=self.dirs_shm.buf)
        self.dirs_shm_mdata = SharedMetaData(self.dirs_shm.name, self.diff_arr_shape, np.byte)
        np.copyto(self.dirs, dirs)

        self.current_count = None
        self.shms_unlinked = False

    def diffuse_bulk(self):
        # mixing particles according to Chopard and Droz
        randomise = np.array(np.random.random_sample(len(self.cells[0])), dtype=np.single)
        # randomise = np.array(np.random.random_sample(len(self.cells[0])))
        # deflection 1
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 1, axis=0)
        # deflection 2
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 1, axis=0)
        self.dirs[:, temp_ind] *= -1
        # deflection 3
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 2, axis=0)
        # deflection 4
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 2, axis=0)
        self.dirs[:, temp_ind] *= -1
        # reflection
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] *= -1

        self.cells = np.add(self.cells, self.dirs, casting="unsafe")

        # adjusting a coordinates of side points for correct shifting
        ind = np.where(self.cells[2] < 0)[0]
        # closed left bound (reflection)
        self.cells[2, ind] = 1
        self.dirs[2, ind] = 1
        # _______________________
        # periodic____________________________________
        # self.cells[2, ind] = self.cells_per_axis - 1
        # ____________________________________________
        # open left bound___________________________
        # self.cells = np.delete(self.cells, ind, 1)
        # self.dirs = np.delete(self.dirs, ind, 1)
        # __________________________________________

        self.cells[0, np.where(self.cells[0] == -1)] = self.cells_per_axis - 1
        self.cells[0, np.where(self.cells[0] == self.cells_per_axis)] = 0
        self.cells[1, np.where(self.cells[1] == -1)] = self.cells_per_axis - 1
        self.cells[1, np.where(self.cells[1] == self.cells_per_axis)] = 0

        ind = np.where(self.cells[2] == self.cells_per_axis)[0]
        # closed right bound (reflection)____________
        # self.cells[2, ind] = self.cells_per_axis - 2
        # self.dirs[2, ind] = -1
        # ___________________________________________
        # open right bound___________________________
        self.cells = np.delete(self.cells, ind, 1)
        self.dirs = np.delete(self.dirs, ind, 1)
        # ___________________________________________
        # periodic____________________________________
        # self.cells[2, ind] = 0
        # ____________________________________________
        self.fill_first_page()

    def diffuse_with_scale(self):
        """
        Outward diffusion through bulk + scale.
        """
        # Diffusion through the scale. If the current particle is inside the product particle
        # it will be reflected
        out_scale = check_in_scale(self.scale.full_c3d, self.cells, self.dirs)

        # Diffusion along grain boundaries
        # ______________________________________________________________________________________________________________
        # exists = self.microstructure.grain_boundaries[self.cells[0], self.cells[1], self.cells[2]]
        # # # print(exists)
        # temp_ind = np.array(np.where(exists)[0], dtype=np.uint32)
        # # print(temp_ind)

        # exists = self.microstructure.grain_boundaries[self.cells[0], self.cells[1], self.cells[2]]
        # # # print(exists)
        # temp_ind = np.array(np.where(exists)[0], dtype=np.uint32)
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
        # cross_shifts = np.array(np.random.choice([0, 1, 2, 3], len(shift_vector[0])), dtype=np.ubyte)
        # cross_shifts = np.array(self.cross_shifts[cross_shifts], dtype=np.byte).transpose()
        #
        # shift_vector += cross_shifts
        #
        # self.cells[:, temp_ind] += shift_vector
        # # print(self.cells)
        # ______________________________________________________________________________________________________________

        # mixing particles according to Chopard and Droz
        randomise = np.array(np.random.random_sample(out_scale.size), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] = np.roll(self.dirs[:, out_scale[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] = np.roll(self.dirs[:, out_scale[temp_ind]], 1, axis=0)
        self.dirs[:, out_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] = np.roll(self.dirs[:, out_scale[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] = np.roll(self.dirs[:, out_scale[temp_ind]], 2, axis=0)
        self.dirs[:, out_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] *= -1

        self.cells = np.add(self.cells, self.dirs, casting="unsafe")
        # adjusting a coordinates of side points for correct shifting
        ind = np.where(self.cells[2] < 0)[0]
        # closed left bound (reflection)
        self.cells[2, ind] = 1
        self.dirs[2, ind] = 1
        # _______________________
        # periodic____________________________________
        # self.cells[2, ind] = self.cells_per_axis - 1
        # ____________________________________________
        # open left bound___________________________
        # self.cells = np.delete(self.cells, ind, 1)
        # self.dirs = np.delete(self.dirs, ind, 1)
        # __________________________________________

        self.cells[0, np.where(self.cells[0] <= -1)] = self.cells_per_axis - 1
        self.cells[0, np.where(self.cells[0] >= self.cells_per_axis)] = 0
        self.cells[1, np.where(self.cells[1] <= -1)] = self.cells_per_axis - 1
        self.cells[1, np.where(self.cells[1] >= self.cells_per_axis)] = 0

        ind = np.where(self.cells[2] >= self.cells_per_axis)[0]
        # closed right bound (reflection)____________
        # self.cells[2, ind] = self.cells_per_axis - 2
        # self.dirs[2, ind] = -1
        # ___________________________________________
        # open right bound___________________________
        self.cells = np.delete(self.cells, ind, 1)
        self.dirs = np.delete(self.dirs, ind, 1)
        # ___________________________________________
        # periodic____________________________________
        # self.cells[2, ind] = 0
        # ____________________________________________
        self.fill_first_page()

    def fill_first_page(self):
        # generating new particles on the diffusion surface (X = self.n_cells_per_axis)
        self.current_count = len(np.where(self.cells[2, :self.last_in_diff_arr] == self.cells_per_axis - 1)[0])
        cells_numb_diff = self.n_per_page - self.current_count
        if cells_numb_diff > 0:
            new_out_page = np.random.randint(self.cells_per_axis, size=(2, cells_numb_diff), dtype=np.short)
            new_out_page = np.concatenate((new_out_page, np.full((1, cells_numb_diff),
                                                                 self.cells_per_axis - 1, dtype=np.short)))
            # appending new generated particles as a ballistic ones to cells
            self.cells[:, self.last_in_diff_arr:self.last_in_diff_arr + cells_numb_diff] = new_out_page
            # appending new direction vectors to dirs
            new_dirs = np.zeros((3, cells_numb_diff), dtype=np.byte)
            new_dirs[2, :] = -1
            self.dirs[:, self.last_in_diff_arr:self.last_in_diff_arr + cells_numb_diff] = new_dirs
            self.last_in_diff_arr += cells_numb_diff

    def dell_cells_from_diff_arrays(self, ind_to_del):
        to_move = np.delete(self.cells[:, :self.last_in_diff_arr], ind_to_del, axis=1)
        self.cells[:, :to_move.shape[1]] = to_move
        to_move = np.delete(self.dirs[:, :self.last_in_diff_arr], ind_to_del, axis=1)
        self.dirs[:, :to_move.shape[1]] = to_move
        self.last_in_diff_arr = to_move.shape[1]

    def get_cells_coords(self):
        return self.cells[:, :self.last_in_diff_arr]

    def transform_to_3d(self, furthest_i):
        if furthest_i + 1 + self.precip_transform_depth + self.neigh_range > self.cells_per_axis:
            last_i = self.cells_per_axis
        else:
            depth = furthest_i + 1 + self.precip_transform_depth + 1
            last_i = depth - 1

        self.i_ind = np.array(np.where(self.cells[2, :self.last_in_diff_arr] < last_i)[0], dtype=np.uint32)
        self.i_descards = np.array(self.cells[:, self.i_ind], dtype=np.short)
        insert_counts(self.c3d, self.i_descards, 1)

        self.in_3D_flag = True

    def transform_to_descards(self):
        ind_out = decrease_counts(self.c3d, self.i_descards)

        to_move = np.delete(self.cells[:, :self.last_in_diff_arr], self.i_ind[ind_out], axis=1)
        self.cells[:, :to_move.shape[1]] = to_move

        to_move = np.delete(self.dirs[:, :self.last_in_diff_arr], self.i_ind[ind_out], axis=1)
        self.dirs[:, :to_move.shape[1]] = to_move

        self.last_in_diff_arr = to_move.shape[1]

        self.in_3D_flag = False

        decomposed = np.array(np.nonzero(self.c3d), dtype=np.short)
        if len(decomposed[0]) > 0:
            counts = self.c3d[decomposed[0], decomposed[1], decomposed[2]]
            decomposed = np.array(np.repeat(decomposed, counts, axis=1), dtype=np.short)
            # self.cells = np.concatenate((self.cells, decomposed), axis=1)

            self.cells[:, self.last_in_diff_arr:self.last_in_diff_arr + decomposed.shape[1]] = decomposed

            new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(decomposed[0]))
            new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
            new_dirs -= 1
            # self.dirs = np.concatenate((self.dirs, new_dirs), axis=1)

            self.dirs[:, self.last_in_diff_arr:self.last_in_diff_arr + decomposed.shape[1]] = new_dirs

            self.last_in_diff_arr += decomposed.shape[1]
            self.c3d[:] = 0

    def count_cells_at_index(self, index):
        return len(np.where(self.cells[2, :self.last_in_diff_arr] == index)[0])

    def close_and_unlink_shm(self):
        if not self.shms_unlinked:
            self.c3d_shared.close()
            self.c3d_shared.unlink()
            self.cells_shm.close()
            self.cells_shm.unlink()
            self.dirs_shm.close()
            self.dirs_shm.unlink()
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
        self.i_descards = None
        self.i_ind = None

        self.p_ranges_scale = self.generate_prob_ranges(settings.PROBABILITIES_SCALE)
        self.p_ranges_interface = self.generate_prob_ranges(settings.PROBABILITIES_INTERFACE)

        self.extended_axis = self.cells_per_axis + self.neigh_range
        self.extended_shape = (self.cells_per_axis, self.cells_per_axis, self.extended_axis)

        temp = np.full(self.extended_shape, 0, dtype=np.ubyte)
        self.c3d_shared = shared_memory.SharedMemory(create=True, size=temp.nbytes)
        self.c3d = np.ndarray(self.extended_shape, dtype=np.ubyte, buffer=self.c3d_shared.buf)

        self.c3d_shm_mdata = SharedMetaData(self.c3d_shared.name, self.extended_shape, np.ubyte)
        self.shms_unlinked = False

        self.scale = None
        self.diffuse = None
        self.n_boost_steps = Config.N_BOOST_STEPS

        self.utils = utils

        self.cells = np.array([[], [], []], dtype=np.short)

        # self.dirs = np.zeros((3, len(self.cells[0])), dtype=np.byte)
        # self.dirs[2] = 1

        dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(self.cells[0]))
        dirs = np.array(np.unravel_index(dirs, (3, 3, 3)), dtype=np.byte)
        dirs -= 1
        self.dirs = dirs

        self.current_count = 0
        self.fill_first_page()

        self.microstructure = None

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
        randomise = np.array(np.random.random_sample(len(self.cells[0])), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 2, axis=0)
        self.dirs[:, temp_ind] *= -1
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] *= -1
        self.cells = np.add(self.cells, self.dirs, casting="unsafe")
        # adjusting a coordinates of side points for correct shifting
        ind = np.where(self.cells[2] < 0)[0]
        # closed left bound (reflection)
        # self.cells[2, ind] = 0
        # self.dirs[2, ind] = 1
        # _______________________
        # open left bound___________________________
        self.cells = np.delete(self.cells, ind, 1)
        self.dirs = np.delete(self.dirs, ind, 1)
        # __________________________________________
        self.cells[0, np.where(self.cells[0] <= -1)] = self.cells_per_axis - 1
        self.cells[0, np.where(self.cells[0] >= self.cells_per_axis)] = 0
        self.cells[1, np.where(self.cells[1] <= -1)] = self.cells_per_axis - 1
        self.cells[1, np.where(self.cells[1] >= self.cells_per_axis)] = 0
        ind = np.where(self.cells[2] >= self.cells_per_axis)
        # closed right bound (reflection)____________
        # self.cells[2, ind] = self.cells_per_axis - 2
        # self.dirs[2, ind] = -1
        # ___________________________________________
        # open right bound___________________________
        self.cells = np.delete(self.cells, ind, 1)
        self.dirs = np.delete(self.dirs, ind, 1)
        # ___________________________________________
        self.current_count = len(np.where(self.cells[2] == 0)[0])
        self.fill_first_page()

    def diffuse_gb(self):
        """
        Inward diffusion through bulk and along grain boundaries.
        """
        # Diffusion along grain boundaries
        # ______________________________________________________________________________________________________________
        exists = self.microstructure.grain_boundaries[self.cells[0], self.cells[1], self.cells[2]]
        t_ind_in_gb, ind_out_gb = separate_in_gb(exists)

        randomise = np.array(np.random.random_sample(len(t_ind_in_gb)), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p0_2d)[0], dtype=np.uint32)

        ind_in_gb = t_ind_in_gb[temp_ind]
        temp_ = np.delete(t_ind_in_gb, temp_ind)

        ind_out_gb = np.concatenate((ind_out_gb, temp_))
        in_gb = np.array(self.cells[:, ind_in_gb], dtype=np.short)

        boost_vector = np.array(self.microstructure.jump_directions[in_gb[0], in_gb[1], in_gb[2]],
                                dtype=np.short).transpose()
        # cross_shifts = np.array(np.random.choice([0, 1, 2, 3], len(ind_in_gb)), dtype=np.ubyte)
        # cross_shifts = np.array(self.cross_shifts[cross_shifts], dtype=np.byte).transpose()

        # boost_vector += cross_shifts
        # self.cells[:, temp_] += cross_shifts
        self.cells[:, ind_in_gb] += boost_vector

        # Diffusion in bulk
        # ______________________________________________________________________________________________________________
        randomise = np.array(np.random.random_sample(len(ind_out_gb)), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        self.dirs[:, ind_out_gb[temp_ind]] = np.roll(self.dirs[:, ind_out_gb[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        self.dirs[:, ind_out_gb[temp_ind]] = np.roll(self.dirs[:, ind_out_gb[temp_ind]], 1, axis=0)
        self.dirs[:, ind_out_gb[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        self.dirs[:, ind_out_gb[temp_ind]] = np.roll(self.dirs[:, ind_out_gb[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        self.dirs[:, ind_out_gb[temp_ind]] = np.roll(self.dirs[:, ind_out_gb[temp_ind]], 2, axis=0)
        self.dirs[:, ind_out_gb[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        self.dirs[:, ind_out_gb[temp_ind]] *= -1

        self.cells = np.add(self.cells, self.dirs, casting="unsafe")
        # adjusting a coordinates of side points for correct shifting
        ind = np.where(self.cells[2] < 0)[0]
        # closed left bound (reflection)
        # self.cells[2, ind] = 0
        # self.dirs[2, ind] = 1
        # _______________________
        # open left bound___________________________
        self.cells = np.delete(self.cells, ind, 1)
        self.dirs = np.delete(self.dirs, ind, 1)
        # __________________________________________

        self.cells[0, np.where(self.cells[0] <= -1)] = self.cells_per_axis - 1
        self.cells[0, np.where(self.cells[0] >= self.cells_per_axis)] = 0
        self.cells[1, np.where(self.cells[1] <= -1)] = self.cells_per_axis - 1
        self.cells[1, np.where(self.cells[1] >= self.cells_per_axis)] = 0

        ind = np.where(self.cells[2] >= self.cells_per_axis)
        # closed right bound (reflection)____________
        # self.cells[2, ind] = self.cells_per_axis - 2
        # self.dirs[2, ind] = -1
        # ___________________________________________
        # open right bound___________________________
        self.cells = np.delete(self.cells, ind, 1)
        self.dirs = np.delete(self.dirs, ind, 1)
        # ___________________________________________

        self.current_count = len(np.where(self.cells[2] == 0)[0])
        self.fill_first_page()

    def diffuse_with_scale(self):
        """
        Inward diffusion through bulk + scale.
        """
        # Diffusion at the interface between matrix the scale. If the current particle is on the product particle
        # it will be boosted along ballistic direction
        self.diffuse_interface()

        # Diffusion through the scale. If the current particle is inside the product particle
        # it will be reflected
        out_scale = check_in_scale(self.scale, self.cells, self.dirs)

        # Diffusion along grain boundaries
        # ______________________________________________________________________________________________________________
        # exists = self.microstructure.grain_boundaries[self.cells[0], self.cells[1], self.cells[2]]
        # # # print(exists)
        # temp_ind = np.array(np.where(exists)[0], dtype=np.uint32)
        # # print(temp_ind)

        # exists = self.microstructure.grain_boundaries[self.cells[0], self.cells[1], self.cells[2]]
        # # # print(exists)
        # temp_ind = np.array(np.where(exists)[0], dtype=np.uint32)
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
        # cross_shifts = np.array(np.random.choice([0, 1, 2, 3], len(shift_vector[0])), dtype=np.ubyte)
        # cross_shifts = np.array(self.cross_shifts[cross_shifts], dtype=np.byte).transpose()
        #
        # shift_vector += cross_shifts
        #
        # self.cells[:, temp_ind] += shift_vector
        # # print(self.cells)
        # ______________________________________________________________________________________________________________

        # mixing particles according to Chopard and Droz
        randomise = np.array(np.random.random_sample(out_scale.size), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] = np.roll(self.dirs[:, out_scale[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] = np.roll(self.dirs[:, out_scale[temp_ind]], 1, axis=0)
        self.dirs[:, out_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] = np.roll(self.dirs[:, out_scale[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] = np.roll(self.dirs[:, out_scale[temp_ind]], 2, axis=0)
        self.dirs[:, out_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] *= -1

        self.cells = np.add(self.cells, self.dirs, casting="unsafe")
        # adjusting a coordinates of side points for correct shifting
        ind = np.where(self.cells[2] < 0)[0]
        # closed left bound (reflection)_______________________
        # self.cells[2, ind] = 0
        # self.dirs[2, ind] = 1
        # _____________________________________________________
        # open left bound___________________________
        self.cells = np.delete(self.cells, ind, 1)
        self.dirs = np.delete(self.dirs, ind, 1)
        # __________________________________________
        # periodic left bound____________________________________
        # self.cells[2, ind] = self.cells_per_axis - 1
        # _______________________________________________________

        self.cells[0, np.where(self.cells[0] <= -1)] = self.cells_per_axis - 1
        self.cells[0, np.where(self.cells[0] >= self.cells_per_axis)] = 0
        self.cells[1, np.where(self.cells[1] <= -1)] = self.cells_per_axis - 1
        self.cells[1, np.where(self.cells[1] >= self.cells_per_axis)] = 0

        ind = np.where(self.cells[2] >= self.cells_per_axis)[0]

        # closed right bound (reflection)____________
        # self.cells[2, ind] = self.cells_per_axis - 2
        # self.dirs[2, ind] = -1
        # ___________________________________________
        # open right bound___________________________
        self.cells = np.delete(self.cells, ind, 1)
        self.dirs = np.delete(self.dirs, ind, 1)
        # ___________________________________________
        # periodic right bound____________________________________
        # self.cells[2, ind] = 0
        # ________________________________________________________

        # ___________________________________
        self.current_count = len(np.where(self.cells[2] == 0)[0])
        self.fill_first_page()
        # ___________________________________

    def diffuse_with_scale_adj(self, time=0):
        """
        Inward diffusion through bulk + scale with P.
        """
        # Diffusion at the interface between matrix the scale. If the current particle is on the product particle
        # it will be boosted along ballistic direction
        # self.diffuse_interface()

        # Diffusion through the scale. If the current particle is inside the product particle
        # it will be reflected
        out_scale, in_scale = check_in_scale_adj(self.scale, self.cells)

        # Diffusion along grain boundaries
        # ______________________________________________________________________________________________________________
        # exists = self.microstructure.grain_boundaries[self.cells[0], self.cells[1], self.cells[2]]
        # # # print(exists)
        # temp_ind = np.array(np.where(exists)[0], dtype=np.uint32)
        # # print(temp_ind)

        # exists = self.microstructure.grain_boundaries[self.cells[0], self.cells[1], self.cells[2]]
        # # # print(exists)
        # temp_ind = np.array(np.where(exists)[0], dtype=np.uint32)
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
        # cross_shifts = np.array(np.random.choice([0, 1, 2, 3], len(shift_vector[0])), dtype=np.ubyte)
        # cross_shifts = np.array(self.cross_shifts[cross_shifts], dtype=np.byte).transpose()
        #
        # shift_vector += cross_shifts
        #
        # self.cells[:, temp_ind] += shift_vector
        # # print(self.cells)
        # ______________________________________________________________________________________________________________

        # mixing particles according to Chopard and Droz
        randomise = np.array(np.random.random_sample(out_scale.size), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] = np.roll(self.dirs[:, out_scale[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] = np.roll(self.dirs[:, out_scale[temp_ind]], 1, axis=0)
        self.dirs[:, out_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] = np.roll(self.dirs[:, out_scale[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] = np.roll(self.dirs[:, out_scale[temp_ind]], 2, axis=0)
        self.dirs[:, out_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        self.dirs[:, out_scale[temp_ind]] *= -1

        # IN Scale Diffusion
        randomise = np.array(np.random.random_sample(in_scale.size), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p_ranges_scale.p1_range)[0], dtype=np.uint32)
        self.dirs[:, in_scale[temp_ind]] = np.roll(self.dirs[:, in_scale[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p1_range) &
                                     (randomise <= self.p_ranges_scale.p2_range))[0], dtype=np.uint32)
        self.dirs[:, in_scale[temp_ind]] = np.roll(self.dirs[:, in_scale[temp_ind]], 1, axis=0)
        self.dirs[:, in_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p2_range) &
                                     (randomise <= self.p_ranges_scale.p3_range))[0], dtype=np.uint32)
        self.dirs[:, in_scale[temp_ind]] = np.roll(self.dirs[:, in_scale[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p3_range) &
                                     (randomise <= self.p_ranges_scale.p4_range))[0], dtype=np.uint32)
        self.dirs[:, in_scale[temp_ind]] = np.roll(self.dirs[:, in_scale[temp_ind]], 2, axis=0)
        self.dirs[:, in_scale[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p4_range) &
                                     (randomise <= self.p_ranges_scale.p_r_range))[0], dtype=np.uint32)
        self.dirs[:, in_scale[temp_ind]] *= -1

        self.cells = np.add(self.cells, self.dirs, casting="unsafe")
        # adjusting a coordinates of side points for correct shifting
        ind = np.where(self.cells[2] < 0)[0]
        # closed left bound (reflection)_______________________
        # self.cells[2, ind] = 0
        # self.dirs[2, ind] = 1
        # _____________________________________________________
        # open left bound___________________________
        self.cells = np.delete(self.cells, ind, 1)
        self.dirs = np.delete(self.dirs, ind, 1)
        # __________________________________________
        # periodic left bound____________________________________
        # self.cells[2, ind] = self.cells_per_axis - 1
        # _______________________________________________________

        self.cells[0, np.where(self.cells[0] <= -1)] = self.cells_per_axis - 1
        self.cells[0, np.where(self.cells[0] >= self.cells_per_axis)] = 0
        self.cells[1, np.where(self.cells[1] <= -1)] = self.cells_per_axis - 1
        self.cells[1, np.where(self.cells[1] >= self.cells_per_axis)] = 0

        ind = np.where(self.cells[2] >= self.cells_per_axis)[0]

        # closed right bound (reflection)____________
        # self.cells[2, ind] = self.cells_per_axis - 2
        # self.dirs[2, ind] = -1
        # ___________________________________________
        # open right bound___________________________
        self.cells = np.delete(self.cells, ind, 1)
        self.dirs = np.delete(self.dirs, ind, 1)
        # ___________________________________________
        # periodic right bound____________________________________
        # self.cells[2, ind] = 0
        # ________________________________________________________

        # ___________________________________
        self.current_count = len(np.where(self.cells[2] == 0)[0])
        self.fill_first_page(time=time)
        # ___________________________________

    def diffuse_interface(self):
        """
        Inward diffusion along the phase interfaces (between matrix and primary product).
        If the current particle has at least one product particle in its flat neighbourhood and no product ahead
        (in its ballistic direction) it will be boosted forwardly in n_boost_steps.
        """
        all_arounds = self.utils.calc_sur_ind_interface(self.cells, self.dirs, self.extended_axis - 1)
        neighbours = go_around_bool(self.scale, all_arounds)
        to_boost = np.array([sum(n_arr[:-1]) * (not n_arr[-1]) for n_arr in neighbours])
        to_boost = np.array(np.where(to_boost)[0])

        if len(to_boost) > 0:
            for _ in range(self.n_boost_steps):
                self.cells[:, to_boost] = np.add(self.cells[:, to_boost], self.dirs[:, to_boost], casting="unsafe")
            # adjusting a coordinates of side points for correct shifting
            self.cells[0, to_boost[np.where(self.cells[0, to_boost] <= -1)]] = self.cells_per_axis - 1
            self.cells[0, to_boost[np.where(self.cells[0, to_boost] >= self.cells_per_axis)]] = 0
            self.cells[1, to_boost[np.where(self.cells[1, to_boost] <= -1)]] = self.cells_per_axis - 1
            self.cells[1, to_boost[np.where(self.cells[1, to_boost] >= self.cells_per_axis)]] = 0

            ind = np.where(self.cells[2, to_boost] < 0)[0]
            # closed left bound (reflection)
            # self.cells[2, to_boost[ind]] = 0
            # self.dirs[2, to_boost[ind]] = 1
            # _______________________
            # open left bound___________________________
            self.cells = np.delete(self.cells, to_boost[ind], 1)
            self.dirs = np.delete(self.dirs, to_boost[ind], 1)
            # __________________________________________

            ind = np.where(self.cells[2] >= self.cells_per_axis)
            # closed right bound (reflection)____________
            # self.cells[2, ind] = self.cells_per_axis - 2
            # self.dirs[2, ind] = -1
            # ___________________________________________
            # open right bound___________________________
            self.cells = np.delete(self.cells, ind, 1)
            self.dirs = np.delete(self.dirs, ind, 1)
            # ___________________________________________

    def diffuse_interface_adj(self):
        """
        Inward diffusion along the phase interfaces (between matrix and primary product).
        If the current particle has at least one product particle in its flat neighbourhood and no product ahead
        (in its ballistic direction) it will be boosted forwardly with higer P.
        """
        all_arounds = self.utils.calc_sur_ind_interface_adj(self.cells, self.dirs, self.extended_axis - 1)
        in_int, blocked, out_int  = separate_in_interface(self.scale, all_arounds)

        # Diffusion along grain boundaries
        # ______________________________________________________________________________________________________________
        # exists = self.microstructure.grain_boundaries[self.cells[0], self.cells[1], self.cells[2]]
        # # # print(exists)
        # temp_ind = np.array(np.where(exists)[0], dtype=np.uint32)
        # # print(temp_ind)

        # exists = self.microstructure.grain_boundaries[self.cells[0], self.cells[1], self.cells[2]]
        # # # print(exists)
        # temp_ind = np.array(np.where(exists)[0], dtype=np.uint32)
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
        # cross_shifts = np.array(np.random.choice([0, 1, 2, 3], len(shift_vector[0])), dtype=np.ubyte)
        # cross_shifts = np.array(self.cross_shifts[cross_shifts], dtype=np.byte).transpose()
        #
        # shift_vector += cross_shifts
        #
        # self.cells[:, temp_ind] += shift_vector
        # # print(self.cells)
        # ______________________________________________________________________________________________________________

        # mixing particles according to Chopard and Droz
        randomise = np.array(np.random.random_sample(out_int.size), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        self.dirs[:, out_int[temp_ind]] = np.roll(self.dirs[:, out_int[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        self.dirs[:, out_int[temp_ind]] = np.roll(self.dirs[:, out_int[temp_ind]], 1, axis=0)
        self.dirs[:, out_int[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        self.dirs[:, out_int[temp_ind]] = np.roll(self.dirs[:, out_int[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        self.dirs[:, out_int[temp_ind]] = np.roll(self.dirs[:, out_int[temp_ind]], 2, axis=0)
        self.dirs[:, out_int[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        self.dirs[:, out_int[temp_ind]] *= -1

        # INTERFACE Diffusion
        randomise = np.array(np.random.random_sample(in_int.size), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p_ranges_interface.p1_range)[0], dtype=np.uint32)
        self.dirs[:, in_int[temp_ind]] = np.roll(self.dirs[:, in_int[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p_ranges_interface.p1_range) & (randomise <= self.p_ranges_interface.p2_range))[0], dtype=np.uint32)
        self.dirs[:, in_int[temp_ind]] = np.roll(self.dirs[:, in_int[temp_ind]], 1, axis=0)
        self.dirs[:, in_int[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p_ranges_interface.p2_range) & (randomise <= self.p_ranges_interface.p3_range))[0], dtype=np.uint32)
        self.dirs[:, in_int[temp_ind]] = np.roll(self.dirs[:, in_int[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p_ranges_interface.p3_range) & (randomise <= self.p_ranges_interface.p4_range))[0], dtype=np.uint32)
        self.dirs[:, in_int[temp_ind]] = np.roll(self.dirs[:, in_int[temp_ind]], 2, axis=0)
        self.dirs[:, in_int[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p_ranges_interface.p4_range) & (randomise <= self.p_ranges_interface.p_r_range))[0], dtype=np.uint32)
        self.dirs[:, in_int[temp_ind]] *= -1

        # IN scale Diffusion
        randomise = np.array(np.random.random_sample(blocked.size), dtype=np.single)
        temp_ind = np.array(np.where(randomise <= self.p_ranges_scale.p1_range)[0], dtype=np.uint32)
        self.dirs[:, blocked[temp_ind]] = np.roll(self.dirs[:, blocked[temp_ind]], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p1_range) & (randomise <= self.p_ranges_scale.p2_range))[0], dtype=np.uint32)
        self.dirs[:, blocked[temp_ind]] = np.roll(self.dirs[:, blocked[temp_ind]], 1, axis=0)
        self.dirs[:, blocked[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p2_range) & (randomise <= self.p_ranges_scale.p3_range))[0], dtype=np.uint32)
        self.dirs[:, blocked[temp_ind]] = np.roll(self.dirs[:, blocked[temp_ind]], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p3_range) & (randomise <= self.p_ranges_scale.p4_range))[0], dtype=np.uint32)
        self.dirs[:, blocked[temp_ind]] = np.roll(self.dirs[:, blocked[temp_ind]], 2, axis=0)
        self.dirs[:, blocked[temp_ind]] *= -1
        temp_ind = np.array(np.where((randomise > self.p_ranges_scale.p4_range) & (randomise <= self.p_ranges_scale.p_r_range))[0], dtype=np.uint32)
        self.dirs[:, blocked[temp_ind]] *= -1

        self.cells = np.add(self.cells, self.dirs, casting="unsafe")
        # adjusting a coordinates of side points for correct shifting
        ind = np.where(self.cells[2] < 0)[0]
        # closed left bound (reflection)_______________________
        # self.cells[2, ind] = 0
        # self.dirs[2, ind] = 1
        # _____________________________________________________
        # open left bound___________________________
        self.cells = np.delete(self.cells, ind, 1)
        self.dirs = np.delete(self.dirs, ind, 1)
        # __________________________________________
        # periodic left bound____________________________________
        # self.cells[2, ind] = self.cells_per_axis - 1
        # _______________________________________________________

        self.cells[0, np.where(self.cells[0] <= -1)] = self.cells_per_axis - 1
        self.cells[0, np.where(self.cells[0] >= self.cells_per_axis)] = 0
        self.cells[1, np.where(self.cells[1] <= -1)] = self.cells_per_axis - 1
        self.cells[1, np.where(self.cells[1] >= self.cells_per_axis)] = 0

        ind = np.where(self.cells[2] >= self.cells_per_axis)[0]

        # closed right bound (reflection)____________
        # self.cells[2, ind] = self.cells_per_axis - 2
        # self.dirs[2, ind] = -1
        # ___________________________________________
        # open right bound___________________________
        self.cells = np.delete(self.cells, ind, 1)
        self.dirs = np.delete(self.dirs, ind, 1)
        # ___________________________________________
        # periodic right bound____________________________________
        # self.cells[2, ind] = 0
        # ________________________________________________________

        # ___________________________________
        self.current_count = len(np.where(self.cells[2] == 0)[0])
        self.fill_first_page()
        # ___________________________________

    def fill_first_page(self):
        # generating new particles on the diffusion surface (X = 0)
        adj_cells_pro_page = self.n_per_page - self.current_count
        if adj_cells_pro_page > 0:
            new_in_page = np.random.randint(self.cells_per_axis, size=(2, adj_cells_pro_page), dtype=np.short)
            new_in_page = np.concatenate((new_in_page, np.zeros((1, adj_cells_pro_page), dtype=np.short)))
            # appending new generated particles as a ballistic ones to cells1
            self.cells = np.concatenate((self.cells, new_in_page), axis=1)
            # appending new direction vectors to dirs
            # new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], adj_cells_pro_page)
            # new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
            # new_dirs -= 1
            new_dirs = np.zeros((3, adj_cells_pro_page), dtype=np.byte)
            new_dirs[2, :] = 1
            self.dirs = np.concatenate((self.dirs, new_dirs), axis=1)

    def transform_to_3d(self):
        insert_counts(self.c3d, self.cells, 1)

    def transform_to_descards(self):
        ind_out = decrease_counts(self.c3d, self.cells)
        self.cells = np.delete(self.cells, ind_out, 1)
        self.dirs = np.delete(self.dirs, ind_out, 1)

    def count_cells_at_index(self, index):
        return len(np.where(self.cells[2] == index)[0])

    def calc_furthest_index(self):
        return np.amax(self.cells[2], initial=-1)

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
            self.c3d_shared.close()
            self.c3d_shared.unlink()
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
