import multiprocessing
import numpy as np
import time
from multiprocessing import shared_memory

N_CELLS = 2001
N_PROC = 10


def check_intersections_single_int(seeds, shm_o, shm_a, shm_p, type):
    shape = (N_CELLS, N_CELLS, 3)

    active = np.ndarray(shape, dtype=type, buffer=shm_a.buf)
    oxidant = np.ndarray(shape, dtype=type, buffer=shm_o.buf)
    product = np.ndarray(shape, dtype=type, buffer=shm_p.buf)

    plane_x_ind = 1
    all_arounds = calc_sur_ind_formation(seeds)
    neighbours = np.array([[[oxidant[point[0], point[1], point[2]] for point in seed_arrounds]]
                           for seed_arrounds in all_arounds], dtype=bool)
    arr_len_in = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)
    index_in = np.where(arr_len_in >= 1)[0]
    if len(index_in) > 0:
        seeds = seeds[index_in]
        neighbours = neighbours[index_in]
        all_arounds = all_arounds[index_in]
        in_to_del = np.array(np.where(neighbours))
        start_seed_index = np.unique(in_to_del[0], return_index=True)[1]
        to_del = np.array([in_to_del[1:, indx:indx + 1] for indx in start_seed_index],
                          dtype=np.ubyte)
        coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del[:, 1])],
                         dtype=np.short)
        coord = np.reshape(coord, (len(coord) * 1, 3)).transpose()
        from_which = np.reshape(to_del[:, 0], (1 * len(to_del)))
        seeds = seeds.transpose()
        product[seeds[0], seeds[1], plane_x_ind] += 1
        active[seeds[0], seeds[1], seeds[2]] -= 1

        temp_ind = np.where(from_which == 0)[0]
        if temp_ind.size != 0:
            elem = coord[:, temp_ind]
            oxidant[elem[0], elem[1], elem[2]] -= 1


def calc_sur_ind_formation(seeds):
    """
    Calculating the descarts surrounding coordinates for each seed excluding the position of the seed itself.
    :param seeds: seeds in descarts coordinates
    :return: around_seeds: array of the surrounding coordinates for each seed (26 flat coordinates for each seed)
    """
    ind_ultimate = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1],  # 5 flat
                             [1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],  # 9  corners
                             [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1],  # 13
                             [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],  # 19 side corners
                             [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=int)
    n_cells_per_axis = N_CELLS
    # generating a neighbouring coordinates for each seed (including the position of the seed itself)
    around_seeds = np.array([[item + ind_ultimate] for item in seeds], dtype=int)[:, 0]
    # applying periodic boundary conditions
    around_seeds[around_seeds == n_cells_per_axis] = 0
    around_seeds[around_seeds == -1] = n_cells_per_axis - 1
    return around_seeds


def generate_fetch_ind():
    length = int((n_cells_per_axis / 3) ** 2)
    fetch_ind = np.zeros((9, 2, length), dtype=int)
    iter_shifts = np.array(np.where(np.ones((3, 3)) == 1)).transpose()
    dummy_grid = np.full((n_cells_per_axis, n_cells_per_axis), True)
    all_coord = np.array(np.nonzero(dummy_grid), dtype=int)
    for step, t in enumerate(iter_shifts):
        t_ind = np.where(((all_coord[0] - t[1]) % 3 == 0) & ((all_coord[1] - t[0]) % 3 == 0))[0]
        fetch_ind[step] = all_coord[:, t_ind]
    return fetch_ind


if __name__ == '__main__':
    # ind_ultimate = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1],  # 5 flat
    #                          [1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],  # 9  corners
    #                          [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1],  # 13
    #                          [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],  # 19 side corners
    #                          [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=int)
    n_cells_per_axis = N_CELLS
    cut_shape = (n_cells_per_axis, n_cells_per_axis, 3)

    product = np.zeros(cut_shape, dtype=np.ubyte)
    active = np.random.randint(2, size=cut_shape, dtype=np.ubyte)
    oxidant = np.random.randint(2, size=cut_shape, dtype=np.ubyte)

    fetch_i = generate_fetch_ind()
    fetch_i = fetch_i[0]

    seeds_out = active[fetch_i[0], fetch_i[1], 1]
    seeds_out = fetch_i[:, np.nonzero(seeds_out)[0]]
    seeds_out = np.vstack((seeds_out, np.ones(len(seeds_out[0]), dtype=np.short)))
    seeds_out = seeds_out.transpose()

    leng = len(seeds_out)
    div = int(leng / N_PROC)
    div_ind = np.arange(div, leng - div, div)
    seeds_out = np.vsplit(seeds_out, div_ind)

    shm_a = shared_memory.SharedMemory(create=True, size=active.nbytes)
    active_s = np.ndarray(active.shape, dtype=active.dtype, buffer=shm_a.buf)
    active_s[:] = active[:]

    shm_o = shared_memory.SharedMemory(create=True, size=oxidant.nbytes)
    oxidant_s = np.ndarray(oxidant.shape, dtype=oxidant.dtype, buffer=shm_o.buf)
    oxidant_s[:] = oxidant[:]

    shm_p = shared_memory.SharedMemory(create=True, size=product.nbytes)
    product_s = np.ndarray(product.shape, dtype=product.dtype, buffer=shm_p.buf)
    product_s[:] = product[:]

    processes = []
    for i in range(len(seeds_out)):
        process = multiprocessing.Process(target=check_intersections_single_int,
                                          args=(seeds_out[i], shm_o, shm_a, shm_p, active.dtype,))
        processes.append(process)

    b = time.time()

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    e = time.time()

    print(e - b)

    active_s = np.ndarray(active.shape, dtype=active.dtype, buffer=shm_a.buf)
    oxidant_s = np.ndarray(oxidant.shape, dtype=oxidant.dtype, buffer=shm_o.buf)
    product_s = np.ndarray(product.shape, dtype=product.dtype, buffer=shm_p.buf)

    print()

# b = time.time()
#
# check_intersections_single_int(seeds_out, 0, 0, 0, 0)
#
# e = time.time()
#
# print(e - b)
