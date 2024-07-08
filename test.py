# import multiprocessing
# from multiprocessing import shared_memory
import numpy as np
from configuration import Config


# def generate_fetch_ind(range):
#     range_start = range[0]
#     range_end = range[1]
#     size = 3 + (Config.NEIGH_RANGE - 1) * 2
#     if Config.N_CELLS_PER_AXIS % size == 0:
#         length = int(((range_end - range_start) / size) * (Config.N_CELLS_PER_AXIS / size))
#         fetch_ind = np.zeros((size ** 2, 2, length), dtype=np.short)
#
#         iter_shifts = np.array(np.where(np.ones((size, size)) == 1)).transpose()
#
#         # Create a dummy grid with the specified range in the first dimension
#         dummy_grid = np.full((range_end - range_start, Config.N_CELLS_PER_AXIS), True)
#         all_coord = np.array(np.nonzero(dummy_grid), dtype=np.short)
#         all_coord[0] += range_start  # Adjust the first dimension coordinates to the given range
#
#         for step, t in enumerate(iter_shifts):
#             t_ind = np.where(((all_coord[0] - t[1]) % size == 0) & ((all_coord[1] - t[0]) % size == 0))[0]
#             fetch_ind[step] = all_coord[:, t_ind]
#
#         return np.array(fetch_ind, dtype=np.ushort)

#
# def generate_fetch_ind(my_range):
#     range_start = my_range[0]
#     range_end = my_range[1]
#     size = 3 + (Config.NEIGH_RANGE - 1) * 2
#     if Config.N_CELLS_PER_AXIS % size == 0:
#         iter_shifts = np.array(np.where(np.ones((size, size)) == 1)).transpose()
#
#         # Create a dummy grid with the specified range in the first dimension
#         dummy_grid = np.full((range_end - range_start, Config.N_CELLS_PER_AXIS), True)
#         all_coord = np.array(np.nonzero(dummy_grid), dtype=np.short)
#         all_coord[0] += range_start  # Adjust the first dimension coordinates to the given range
#
#         # Initialize fetch_ind as a list to accommodate varying lengths of coordinate arrays
#         fetch_ind = []
#
#         for step, t in enumerate(iter_shifts):
#             t_ind = np.where(((all_coord[0] - t[1]) % size == 0) & ((all_coord[1] - t[0]) % size == 0))[0]
#             fetch_ind.append(all_coord[:, t_ind])
#
#         # Convert the list of arrays to an object array for easier manipulation later
#         return np.array(fetch_ind, dtype=object)


def generate_fetch_ind_mp(ranges, switch=False):
    size = 3 + (Config.NEIGH_RANGE - 1) * 2
    if Config.N_CELLS_PER_AXIS % size == 0:
        # length = int((Config.N_CELLS_PER_AXIS / size) ** 2)
        # fetch_ind = np.zeros((size**2, 2, length), dtype=np.short)
        iter_shifts = np.array(np.where(np.ones((size, size)) == 1)).transpose()
        dummy_grid = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS), False)
        if switch:
            dummy_grid[ranges[0][0]:ranges[0][1], :] = True
            dummy_grid[ranges[1][0]:ranges[1][1], :] = True
        else:
            dummy_grid[ranges[0]:ranges[1], :] = True
        # dummy_grid[start:end, :] = True
        # dummy_grid[-5:1, :] = True
        n_fetch = []
        all_coord = np.array(np.nonzero(dummy_grid), dtype=np.short)
        for step, t in enumerate(iter_shifts):
            t_ind = np.where(((all_coord[0] - t[1]) % size == 0) & ((all_coord[1] - t[0]) % size == 0))[0]
            # fetch_ind[step] = all_coord[:, t_ind]
            if len(t_ind) > 0:
                n_fetch.append(all_coord[:, t_ind])
        # fetch_ind = np.array(fetch_ind, dtype=np.ushort)

        dummy_grid[:] = False
        for item in n_fetch:
            dummy_grid[item[0], item[1]] = True

    return n_fetch


numb_of_div_per_page = 7

p_chunk_size = int((Config.N_CELLS_PER_AXIS / numb_of_div_per_page) - Config.NEIGH_RANGE * 2)
s_chunk_size = Config.NEIGH_RANGE * 2


p_chunk_ranges = np.zeros((numb_of_div_per_page, 2), dtype=int)
p_chunk_ranges[0] = [Config.NEIGH_RANGE, Config.NEIGH_RANGE + p_chunk_size]

for pos in range(1, numb_of_div_per_page):
    p_chunk_ranges[pos, 0] = p_chunk_ranges[pos-1, 1] + s_chunk_size
    p_chunk_ranges[pos, 1] = p_chunk_ranges[pos, 0] + p_chunk_size

s_chunk_ranges = np.zeros((numb_of_div_per_page + 1, 2), dtype=int)
s_chunk_ranges[0] = [0, Config.NEIGH_RANGE]

for pos in range(1, numb_of_div_per_page+1):
    s_chunk_ranges[pos, 0] = s_chunk_ranges[pos-1, 1] + p_chunk_size
    s_chunk_ranges[pos, 1] = s_chunk_ranges[pos, 0] + s_chunk_size

s_chunk_ranges[-1, 1] = Config.N_CELLS_PER_AXIS



p_ind = []
for item in p_chunk_ranges:
    new_batch = generate_fetch_ind_mp(item)
    p_ind.append(new_batch)

s_ind = []
f_and_l = generate_fetch_ind_mp([s_chunk_ranges[0], s_chunk_ranges[-1]], switch=True)
s_ind.append(f_and_l)
for index, item in enumerate(s_chunk_ranges):
    if index == 0 or index == len(s_chunk_ranges) - 1:
        continue
    new_batch = generate_fetch_ind_mp(item)
    s_ind.append(new_batch)

dummy_grid1 = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS), False)

for item in p_ind:
    for coord_set in item:
        for ind in range(len(coord_set[0])):
            z_coord = coord_set[0, ind]
            y_coord = coord_set[1, ind]
            if dummy_grid1[z_coord, y_coord]:
                print("ALLREADY TRUE AT: ", z_coord, " ", y_coord)
            else:
                dummy_grid1[z_coord, y_coord] = True
print()

dummy_grid2 = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS), False)

for item in s_ind:
    for coord_set in item:
        for ind in range(len(coord_set[0])):
            z_coord = coord_set[0, ind]
            y_coord = coord_set[1, ind]
            if dummy_grid2[z_coord, y_coord]:
                print("ALLREADY TRUE AT: ", z_coord, " ", y_coord)
            else:
                dummy_grid2[z_coord, y_coord] = True
print()

#
# # import utils
# # from configuration import Config
# import numba
# from precompiled_modules import precompiled_numba
#
# SHAPE = (100, 100, 100)
# DTYPE = np.ubyte
#
# N_PROC = 20
# N_TASKS = 10
# N_ITER = 1000000
#
#
# # @numba.njit(nopython=True, fastmath=True)
# # def increase_counts(array_2d):
# #     for ind_x in range(array_2d.shape[0]):
# #         for ind_y in range(array_2d.shape[1]):
# #             array_2d[ind_x, ind_y] += 1
#
#
# class Other:
#     def __init__(self):
#         some_huge_shit = np.zeros(SHAPE, dtype=DTYPE)
#         self.huge_shit_shm = shared_memory.SharedMemory(create=True, size=some_huge_shit.nbytes)
#
#         self.huge_shit = np.ndarray(some_huge_shit.shape, dtype=some_huge_shit.dtype, buffer=self.huge_shit_shm.buf)
#         np.copyto(self.huge_shit, some_huge_shit)
#
#         self.huge_shit_mdata = SharedMetaData(self.huge_shit_shm.name, self.huge_shit.shape, self.huge_shit.dtype)
#
#         self.scale = None
#
#     def do_in_parent(self):
#         precompiled_numba.insert_counts(self.huge_shit, np.array([[0, 0, 0], [1, 1, 1]], dtype=np.short))
#         # increase_counts(self.huge_shit)
#         print("Done In Parent")
#
#
# class SharedMetaData:
#     def __init__(self, shm_name, shape, dtype):
#         self.name = shm_name
#         self.shape = shape
#         self.dtype = dtype
#
#
# def worker(args):
#     callback = args[-1]
#     args = args[:-1]
#     result = callback(*args)
#     return result
#
#
# def heavy_work(shm_mdata, huge_shit_mdata):
#
#     shm_o = shared_memory.SharedMemory(name=shm_mdata.name)
#     my_array = np.ndarray(shm_mdata.shape, dtype=shm_mdata.dtype, buffer=shm_o.buf)
#
#     shm_other = shared_memory.SharedMemory(name=huge_shit_mdata.name)
#     other_array = np.ndarray(huge_shit_mdata.shape, dtype=huge_shit_mdata.dtype, buffer=shm_other.buf)
#
#     precompiled_numba.insert_counts(my_array, np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 2, 3]], dtype=np.short).transpose())
#
#     shm_o.close()
#     shm_other.close()
#     return 0
#
#
# class MyPool:
#     def __init__(self):
#         self.pool = multiprocessing.Pool(processes=N_PROC, maxtasksperchild=100000)
#
#         some_huge_shit = np.zeros(SHAPE, dtype=DTYPE)
#         self.huge_shit_shm = shared_memory.SharedMemory(create=True, size=some_huge_shit.nbytes)
#
#         huge_shit = np.ndarray(some_huge_shit.shape, dtype=some_huge_shit.dtype, buffer=self.huge_shit_shm.buf)
#         np.copyto(huge_shit, some_huge_shit)
#
#         self.huge_shit_mdata = SharedMetaData(self.huge_shit_shm.name, huge_shit.shape, huge_shit.dtype)
#
#         self.other = Other()
#
#     def start_pool(self):
#         tasks = [(self.huge_shit_mdata, self.other.huge_shit_mdata, heavy_work) for _ in range(N_TASKS)]
#         results = self.pool.map(worker, tasks)
#         print("done_pool")
#
#     def terminate_workers(self):
#         # self.pool.close()
#         # self.pool.join()
#
#         self.huge_shit_shm.close()
#         self.huge_shit_shm.unlink()
#
#
# if __name__ == '__main__':
#     new_pool = MyPool()
#
#     for iteration in range(N_ITER):
#         print(iteration)
#         new_pool.start_pool()
#
#         # my_array = np.ndarray(new_pool.huge_shit_mdata.shape, dtype=new_pool.huge_shit_mdata.dtype, buffer=new_pool.huge_shit_shm.buf)
#         # print()
