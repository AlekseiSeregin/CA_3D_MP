import numpy as np
from multiprocessing import shared_memory


def diffuse_bulk_mp(working_range, cells_shm_mdata, dirs_shm_mdata, cells_per_axis, p_ranges):

    size_of_chunk = working_range[1] - working_range[0]
    working_range = np.arange(working_range[0], working_range[1])

    shm_cells = shared_memory.SharedMemory(name=cells_shm_mdata.name)
    cells = np.ndarray(cells_shm_mdata.shape, dtype=cells_shm_mdata.dtype, buffer=shm_cells.buf)

    shm_dirs = shared_memory.SharedMemory(name=dirs_shm_mdata.name)
    dirs = np.ndarray(dirs_shm_mdata.shape, dtype=dirs_shm_mdata.dtype, buffer=shm_dirs.buf)

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
    # self.cells[2, ind_right] = self.cells_per_axis - 2
    # self.dirs[2, ind_right] = -1
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