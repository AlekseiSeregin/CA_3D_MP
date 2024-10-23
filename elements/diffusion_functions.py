import numpy as np
from multiprocessing import shared_memory
from utils import numba_functions


def diffuse_bulk_mp(working_range, cur_case, p_ranges):
    cells_per_axis = cur_case.cells_per_axis
    size_of_chunk = working_range[1] - working_range[0]
    working_range = np.arange(working_range[0], working_range[1])

    shm_cells = shared_memory.SharedMemory(name=cur_case.active_cells_shm_mdata.name)
    cells = np.ndarray(cur_case.active_cells_shm_mdata.shape, dtype=cur_case.active_cells_shm_mdata.dtype, buffer=shm_cells.buf)

    shm_dirs = shared_memory.SharedMemory(name=cur_case.active_dirs_shm_mdata.name)
    dirs = np.ndarray(cur_case.active_dirs_shm_mdata.shape, dtype=cur_case.active_dirs_shm_mdata.dtype, buffer=shm_dirs.buf)

    # mixing particles according to Chopard and Droz
    randomise = np.array(np.random.random_sample(size_of_chunk), dtype=np.single)
    # randomise = np.array(np.random.random_sample(len(self.cells[0])))
    # deflection 1
    temp_ind = np.array(np.where(randomise <= p_ranges.p1_range)[0], dtype=np.uint32)
    dirs[:, working_range[temp_ind]] = np.roll(dirs[:, working_range[temp_ind]], 1, axis=0)
    # deflection 2
    temp_ind = np.array(np.where((randomise > p_ranges.p1_range) & (randomise <= p_ranges.p2_range))[0], dtype=np.uint32)
    dirs[:, working_range[temp_ind]] = np.roll(dirs[:, working_range[temp_ind]], 1, axis=0)
    dirs[:, working_range[temp_ind]] *= -1
    # deflection 3
    temp_ind = np.array(np.where((randomise > p_ranges.p2_range) & (randomise <= p_ranges.p3_range))[0], dtype=np.uint32)
    dirs[:, working_range[temp_ind]] = np.roll(dirs[:, working_range[temp_ind]], 2, axis=0)
    # deflection 4
    temp_ind = np.array(np.where((randomise > p_ranges.p3_range) & (randomise <= p_ranges.p4_range))[0], dtype=np.uint32)
    dirs[:, working_range[temp_ind]] = np.roll(dirs[:, working_range[temp_ind]], 2, axis=0)
    dirs[:, working_range[temp_ind]] *= -1
    # reflection
    temp_ind = np.array(np.where((randomise > p_ranges.p4_range) & (randomise <= p_ranges.p_r_range))[0], dtype=np.uint32)
    dirs[:, working_range[temp_ind]] *= -1

    cells[:, working_range] = np.add(cells[:, working_range], dirs[:, working_range], casting="unsafe")

    # adjusting a coordinates of side points for correct shifting
    ind_left = np.where(cells[2, working_range] < 0)[0]
    # closed left bound (reflection)
    # cells[2, working_range[ind_left]] = 1
    # dirs[2, working_range[ind_left]] = 1
    # ind_left = np.array([], dtype=int)
    # _______________________
    # periodic____________________________________
    # self.cells[2, working_range[ind_left]] = self.cells_per_axis - 1
    # ind_left = []
    # ____________________________________________
    # open left bound___________________________
    # if only ind_left!!!
    # __________________________________________

    cells[0, working_range[np.where(cells[0, working_range] < 0)[0]]] = cells_per_axis - 1
    cells[0, working_range[np.where(cells[0, working_range] > cells_per_axis - 1)[0]]] = 0
    cells[1, working_range[np.where(cells[1, working_range] < 0)[0]]] = cells_per_axis - 1
    cells[1, working_range[np.where(cells[1, working_range] > cells_per_axis - 1)[0]]] = 0

    ind_right = np.where(cells[2, working_range] > cells_per_axis - 1)[0]
    # closed right bound (reflection)____________
    # cells[2, ind_right] = cells_per_axis - 2
    # dirs[2, ind_right] = -1
    # ind_right = []
    # ___________________________________________
    # open right bound___________________________
    # if only ind_right!!!
    # ___________________________________________
    # periodic____________________________________
    # self.cells[2, working_range[ind_right]] = 0
    # ind_right = = np.array([], dtype=int)
    # ____________________________________________

    shm_cells.close()
    shm_dirs.close()

    return working_range[np.concatenate((ind_left, ind_right))]


def diffuse_with_scale(working_range, cur_case, p_ranges):
    """
    Inward diffusion through bulk + scale.
    """
    cells_per_axis = cur_case.cells_per_axis
    # size_of_chunk = working_range[1] - working_range[0]
    working_range = np.arange(working_range[0], working_range[1])

    shm_cells = shared_memory.SharedMemory(name=cur_case.active_cells_shm_mdata.name)
    cells = np.ndarray(cur_case.active_cells_shm_mdata.shape, dtype=cur_case.active_cells_shm_mdata.dtype,
                       buffer=shm_cells.buf)

    shm_dirs = shared_memory.SharedMemory(name=cur_case.active_dirs_shm_mdata.name)
    dirs = np.ndarray(cur_case.active_dirs_shm_mdata.shape, dtype=cur_case.active_dirs_shm_mdata.dtype,
                      buffer=shm_dirs.buf)

    shm_scale = shared_memory.SharedMemory(name=cur_case.to_check_with_shm_mdata.name)
    scale = np.ndarray(cur_case.to_check_with_shm_mdata.shape, dtype=cur_case.to_check_with_shm_mdata.dtype,
                      buffer=shm_scale.buf)

    # Diffusion at the interface between matrix the scale. If the current particle is on the product particle
    # it will be boosted along ballistic direction
    # self.diffuse_interface()

    # Diffusion through the scale. If the current particle is inside the product particle
    # it will be reflected
    out_scale = numba_functions.check_in_scale_mp(scale, cells, dirs, working_range)

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
    temp_ind = np.array(np.where(randomise <= p_ranges.p1_range)[0], dtype=np.uint32)
    dirs[:, working_range[out_scale[temp_ind]]] = np.roll(dirs[:, working_range[out_scale[temp_ind]]], 1, axis=0)
    temp_ind = np.array(np.where((randomise > p_ranges.p1_range) & (randomise <= p_ranges.p2_range))[0], dtype=np.uint32)
    dirs[:, working_range[out_scale[temp_ind]]] = np.roll(dirs[:, working_range[out_scale[temp_ind]]], 1, axis=0)
    dirs[:, working_range[out_scale[temp_ind]]] *= -1
    temp_ind = np.array(np.where((randomise > p_ranges.p2_range) & (randomise <= p_ranges.p3_range))[0], dtype=np.uint32)
    dirs[:, working_range[out_scale[temp_ind]]] = np.roll(dirs[:, working_range[out_scale[temp_ind]]], 2, axis=0)
    temp_ind = np.array(np.where((randomise > p_ranges.p3_range) & (randomise <= p_ranges.p4_range))[0], dtype=np.uint32)
    dirs[:, working_range[out_scale[temp_ind]]] = np.roll(dirs[:, working_range[out_scale[temp_ind]]], 2, axis=0)
    dirs[:, working_range[out_scale[temp_ind]]] *= -1
    temp_ind = np.array(np.where((randomise > p_ranges.p4_range) & (randomise <= p_ranges.p_r_range))[0], dtype=np.uint32)
    dirs[:, working_range[out_scale[temp_ind]]] *= -1

    cells[:, working_range] = np.add(cells[:, working_range], dirs[:, working_range], casting="unsafe")

    # adjusting a coordinates of side points for correct shifting
    ind_left = np.where(cells[2, working_range] < 0)[0]
    # closed left bound (reflection)
    cells[2, working_range[ind_left]] = 1
    dirs[2, working_range[ind_left]] = 1
    ind_left = np.array([], dtype=int)
    # _______________________
    # periodic____________________________________
    # self.cells[2, working_range[ind_left]] = self.cells_per_axis - 1
    # ind_left = []
    # ____________________________________________
    # open left bound___________________________
    # if only ind_left!!!
    # __________________________________________

    cells[0, working_range[np.where(cells[0, working_range] < 0)[0]]] = cells_per_axis - 1
    cells[0, working_range[np.where(cells[0, working_range] > cells_per_axis - 1)[0]]] = 0
    cells[1, working_range[np.where(cells[1, working_range] < 0)[0]]] = cells_per_axis - 1
    cells[1, working_range[np.where(cells[1, working_range] > cells_per_axis - 1)[0]]] = 0

    ind_right = np.where(cells[2, working_range] > cells_per_axis - 1)[0]
    # closed right bound (reflection)____________
    # cells[2, ind_right] = cells_per_axis - 2
    # dirs[2, ind_right] = -1
    # ind_right = []
    # ___________________________________________
    # open right bound___________________________
    # if only ind_right!!!
    # ___________________________________________
    # periodic____________________________________
    # self.cells[2, working_range[ind_right]] = 0
    # ind_right = = np.array([], dtype=int)
    # ____________________________________________

    shm_cells.close()
    shm_dirs.close()

    return working_range[np.concatenate((ind_left, ind_right))]


def diffuse_with_scale_adj(working_range, cur_case, p_ranges):
    """
    Inward diffusion through bulk + scale.
    """
    cells_per_axis = cur_case.cells_per_axis
    # size_of_chunk = working_range[1] - working_range[0]
    working_range = np.arange(working_range[0], working_range[1])

    shm_cells = shared_memory.SharedMemory(name=cur_case.active_cells_shm_mdata.name)
    cells = np.ndarray(cur_case.active_cells_shm_mdata.shape, dtype=cur_case.active_cells_shm_mdata.dtype,
                       buffer=shm_cells.buf)

    shm_dirs = shared_memory.SharedMemory(name=cur_case.active_dirs_shm_mdata.name)
    dirs = np.ndarray(cur_case.active_dirs_shm_mdata.shape, dtype=cur_case.active_dirs_shm_mdata.dtype,
                      buffer=shm_dirs.buf)

    shm_scale = shared_memory.SharedMemory(name=cur_case.to_check_with_shm_mdata.name)
    scale = np.ndarray(cur_case.to_check_with_shm_mdata.shape, dtype=cur_case.to_check_with_shm_mdata.dtype,
                      buffer=shm_scale.buf)

    # Diffusion at the interface between matrix the scale. If the current particle is on the product particle
    # it will be boosted along ballistic direction
    # self.diffuse_interface()

    # Diffusion through the scale. If the current particle is inside the product particle
    # it will be reflected
    out_scale, in_scale = numba_functions.check_in_scale_mp_adj(scale, cells, working_range)

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
    temp_ind = np.array(np.where(randomise <= p_ranges.p1_range)[0], dtype=np.uint32)
    dirs[:, working_range[out_scale[temp_ind]]] = np.roll(dirs[:, working_range[out_scale[temp_ind]]], 1, axis=0)
    temp_ind = np.array(np.where((randomise > p_ranges.p1_range) & (randomise <= p_ranges.p2_range))[0], dtype=np.uint32)
    dirs[:, working_range[out_scale[temp_ind]]] = np.roll(dirs[:, working_range[out_scale[temp_ind]]], 1, axis=0)
    dirs[:, working_range[out_scale[temp_ind]]] *= -1
    temp_ind = np.array(np.where((randomise > p_ranges.p2_range) & (randomise <= p_ranges.p3_range))[0], dtype=np.uint32)
    dirs[:, working_range[out_scale[temp_ind]]] = np.roll(dirs[:, working_range[out_scale[temp_ind]]], 2, axis=0)
    temp_ind = np.array(np.where((randomise > p_ranges.p3_range) & (randomise <= p_ranges.p4_range))[0], dtype=np.uint32)
    dirs[:, working_range[out_scale[temp_ind]]] = np.roll(dirs[:, working_range[out_scale[temp_ind]]], 2, axis=0)
    dirs[:, working_range[out_scale[temp_ind]]] *= -1
    temp_ind = np.array(np.where((randomise > p_ranges.p4_range) & (randomise <= p_ranges.p_r_range))[0], dtype=np.uint32)
    dirs[:, working_range[out_scale[temp_ind]]] *= -1

    # Diffusion in scale
    randomise = np.array(np.random.random_sample(in_scale.size), dtype=np.single)
    temp_ind = np.array(np.where(randomise <= p_ranges.p1_range)[0], dtype=np.uint32)
    dirs[:, working_range[in_scale[temp_ind]]] = np.roll(dirs[:, working_range[in_scale[temp_ind]]], 1, axis=0)
    temp_ind = np.array(np.where((randomise > p_ranges.p1_range) & (randomise <= p_ranges.p2_range))[0],
                        dtype=np.uint32)
    dirs[:, working_range[in_scale[temp_ind]]] = np.roll(dirs[:, working_range[in_scale[temp_ind]]], 1, axis=0)
    dirs[:, working_range[in_scale[temp_ind]]] *= -1
    temp_ind = np.array(np.where((randomise > p_ranges.p2_range) & (randomise <= p_ranges.p3_range))[0],
                        dtype=np.uint32)
    dirs[:, working_range[in_scale[temp_ind]]] = np.roll(dirs[:, working_range[in_scale[temp_ind]]], 2, axis=0)
    temp_ind = np.array(np.where((randomise > p_ranges.p3_range) & (randomise <= p_ranges.p4_range))[0],
                        dtype=np.uint32)
    dirs[:, working_range[in_scale[temp_ind]]] = np.roll(dirs[:, working_range[in_scale[temp_ind]]], 2, axis=0)
    dirs[:, working_range[in_scale[temp_ind]]] *= -1
    temp_ind = np.array(np.where((randomise > p_ranges.p4_range) & (randomise <= p_ranges.p_r_range))[0],
                        dtype=np.uint32)
    dirs[:, working_range[in_scale[temp_ind]]] *= -1

    cells[:, working_range] = np.add(cells[:, working_range], dirs[:, working_range], casting="unsafe")

    # adjusting a coordinates of side points for correct shifting
    ind_left = np.where(cells[2, working_range] < 0)[0]
    # closed left bound (reflection)
    cells[2, working_range[ind_left]] = 1
    dirs[2, working_range[ind_left]] = 1
    ind_left = np.array([], dtype=int)
    # _______________________
    # periodic____________________________________
    # self.cells[2, working_range[ind_left]] = self.cells_per_axis - 1
    # ind_left = []
    # ____________________________________________
    # open left bound___________________________
    # if only ind_left!!!
    # __________________________________________

    cells[0, working_range[np.where(cells[0, working_range] < 0)[0]]] = cells_per_axis - 1
    cells[0, working_range[np.where(cells[0, working_range] > cells_per_axis - 1)[0]]] = 0
    cells[1, working_range[np.where(cells[1, working_range] < 0)[0]]] = cells_per_axis - 1
    cells[1, working_range[np.where(cells[1, working_range] > cells_per_axis - 1)[0]]] = 0

    ind_right = np.where(cells[2, working_range] > cells_per_axis - 1)[0]
    # closed right bound (reflection)____________
    # cells[2, ind_right] = cells_per_axis - 2
    # dirs[2, ind_right] = -1
    # ind_right = []
    # ___________________________________________
    # open right bound___________________________
    # if only ind_right!!!
    # ___________________________________________
    # periodic____________________________________
    # self.cells[2, working_range[ind_right]] = 0
    # ind_right = = np.array([], dtype=int)
    # ____________________________________________

    shm_cells.close()
    shm_dirs.close()

    return working_range[np.concatenate((ind_left, ind_right))]
