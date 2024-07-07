import numpy as np
import numba
import gc
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

user_input = {"start_real_radius": 0.5,
              "delta_radius": 0.1,
              "n_cells_per_axis": 501  # must be odd!!
              }


class Sphere:
    def __init__(self, params):
        self.ind_all = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1],

                                 [1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],
                                 [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1],

                                 [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],
                                 [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]],
                                dtype=np.byte)

        self.aggregated_ind = np.array([[7, 0, 1, 2, 19, 16, 14],
                                        [6, 0, 1, 5, 18, 15, 14],
                                        [8, 0, 4, 5, 20, 15, 17],
                                        [9, 0, 4, 2, 21, 16, 17],
                                        [11, 3, 1, 2, 19, 24, 22],
                                        [10, 3, 1, 5, 18, 23, 22],
                                        [12, 3, 4, 5, 20, 23, 25],
                                        [13, 3, 4, 2, 21, 24, 25]], dtype=np.int64)

        self.ind_flat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                                 dtype=np.byte)

        self.real_radius = params["start_real_radius"]
        self.delta_radius = params["delta_radius"]
        self.n_cells_per_axis = params["n_cells_per_axis"]

        self.s_coord = int((self.n_cells_per_axis - 1) / 2)
        self.curr_side_lim = math.ceil(self.real_radius - 1 / 2)

        self.on_surface = np.empty((self.n_cells_per_axis ** 3, 3), dtype=np.short)
        self.on_surface[0] = [self.s_coord, self.s_coord, self.s_coord]
        self.last_on_surface_ind = 1

        self.shape = (self.n_cells_per_axis, self.n_cells_per_axis, self.n_cells_per_axis)
        self.c3d = np.array(np.full(self.shape, False, dtype=bool))
        self.c3d[self.s_coord, self.s_coord, self.s_coord] = True

        # self.stats = {params["start_real_radius"]: {"mean": 0,
        #                                             "n_cells": 0,
        #                                             0: 0,
        #                                             1: 0,
        #                                             2: 0,
        #                                             3: 0,
        #                                             4: 0,
        #                                             5: 0}}

        self.stats = {params["start_real_radius"]: {
            "NO_Block": {
                "mean": 0,
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0
            },
            "Block": {
                "mean": 0,
                3: 0,
                4: 0,
                5: 0
            },

            "Block_counts": {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
            }
        }}

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.cell_size = 50
        # self.inside = np.empty((self.n_cells_per_axis**3, 3), dtype=np.short)
        # self.last_inside_ind = 0

    def calc_all_sur_coord(self, seeds):
        around_seeds= np.array([item + self.ind_all for item in seeds], dtype=np.short)
        return around_seeds

    def calc_flat_sur_coord(self, seeds):
        around_seeds = np.array([item + self.ind_flat for item in seeds], dtype=np.short)
        return around_seeds

    def expand_radius(self):
        self.real_radius += self.delta_radius
        self.curr_side_lim = math.ceil(self.real_radius - (1 / 2))
        # self.stats[self.real_radius] = {
        #     "NO_Block": {"mean": 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        #     "block": {"mean": 0, 3: 0, 4: 0, 5: 0 }
        #                                 }

    def update_stats(self, mean_in_block, mean_no_block, freq_block, freq_NO_block, unique_numbs_block, unique_numbs_NO_block, freq_block_counts, unique_block_counts):
        self.stats[self.real_radius] = {
            "NO_Block": {
                "mean": 0,
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0
            },
            "Block": {
                "mean": 0,
                3: 0,
                4: 0,
                5: 0
            },
            "Block_counts": {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
            }
        }

        self.stats[self.real_radius]["NO_Block"]["mean"] = mean_no_block
        self.stats[self.real_radius]["Block"]["mean"] = mean_in_block
        # self.stats[self.real_radius]["n_cells"] = int(np.sum(self.c3d[:, :, self.s_coord]))

        for freq_i, number in zip(freq_block, unique_numbs_block):
            self.stats[self.real_radius]["Block"][number] = freq_i

        for freq_i, number in zip(freq_NO_block, unique_numbs_NO_block):
            self.stats[self.real_radius]["NO_Block"][number] = freq_i

        for freq_i, number in zip(freq_block_counts, unique_block_counts):
            self.stats[self.real_radius]["Block_counts"][number] = freq_i

    def calc_mean_neigh(self):
        self.expand_radius()
        arrounds = self.calc_all_sur_coord(self.on_surface[:self.last_on_surface_ind])

        self.last_on_surface_ind = check_coords_if_belong(self.c3d, arrounds, self.real_radius, self.s_coord,
                                                          self.on_surface,
                                                          self.last_on_surface_ind)

        # flat_arounds = self.calc_flat_sur_coord(self.on_surface[:self.last_on_surface_ind])
        all_arounds = self.calc_all_sur_coord(self.on_surface[:self.last_on_surface_ind])

        neighbours = go_around_bool(self.c3d, all_arounds)

        arr_len_flat = np.sum(neighbours[:, :6], axis=1)
        # flat_arr_len = np.array([np.sum(item[:6], axis=1) for item in neighbours], dtype=np.ubyte)

        on_surface_ind = np.array(np.where(arr_len_flat < 6)[0])
        arr_len_flat = arr_len_flat[on_surface_ind]
        neighbours = neighbours[on_surface_ind]

        self.last_on_surface_ind = len(on_surface_ind)
        self.on_surface[:self.last_on_surface_ind] = self.on_surface[on_surface_ind]

        # ind_where_blocks = aggregate(self.aggregated_ind, neighbours)
        block_counts = aggregate_and_count(self.aggregated_ind, neighbours)
        ind_where_blocks = np.where(block_counts)[0]


        unique_block_counts = np.array(np.unique(block_counts, return_counts=True))
        freq_block_counts = np.array(unique_block_counts[1] / np.sum(unique_block_counts[1]))


        arr_len_flat_block = arr_len_flat[ind_where_blocks]
        arr_len_flat_NO_block = np.delete(arr_len_flat, ind_where_blocks)

        # all_arounds = self.calc_all_sur_coord(self.on_surface[:self.last_on_surface_ind])
        # all_neighbours = go_around_bool(self.c3d, all_arounds)
        # all_arr_len = np.array([np.sum(item) for item in all_neighbours], dtype=np.ubyte)
        # all_mean = np.mean(all_arr_len)

        mean_in_block = np.mean(arr_len_flat_block)
        mean_no_block = np.mean(arr_len_flat_NO_block)

        unique_numbs_block = np.array(np.unique(arr_len_flat_block, return_counts=True))
        freq_block = np.array(unique_numbs_block[1] / np.sum(unique_numbs_block[1]))

        unique_numbs_NO_block = np.array(np.unique(arr_len_flat_NO_block, return_counts=True))
        freq_NO_block = np.array(unique_numbs_NO_block[1] / np.sum(unique_numbs_NO_block[1]))

        # print(self.real_radius, unique_numbs_block[0], freq_block, unique_numbs_NO_block[0], freq_NO_block, sep=" ")
        self.update_stats(mean_in_block, mean_no_block,  freq_block, freq_NO_block, unique_numbs_block[0], unique_numbs_NO_block[0], freq_block_counts, unique_block_counts[0])

        # print(self.real_radius, mean, all_mean, sep=" ")
        # return np.mean(on_surface), unique_numbs[0], freq
        # return np.mean(on_surface)

    def plot_on_surface(self):
        self.ax.cla()
        curr_lim = self.curr_side_lim + 5
        self.ax.set_xlim3d(self.s_coord - curr_lim, self.s_coord + curr_lim)
        self.ax.set_ylim3d(self.s_coord - curr_lim, self.s_coord + curr_lim)
        self.ax.set_zlim3d(self.s_coord - curr_lim, self.s_coord + curr_lim)
        self.ax.scatter(self.on_surface[:self.last_on_surface_ind][:, 2],
                        self.on_surface[:self.last_on_surface_ind][:, 1],
                        self.on_surface[:self.last_on_surface_ind][:, 0],
                        marker=',', color='r', s=self.cell_size * (72. / self.fig.dpi) ** 2, edgecolors='black', linewidth=1)
        plt.show()
        # plt.close()
        # plt.savefig(f'W:/SIMCA/test_runs_data/{self.real_radius}.jpeg')


ind_formation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.byte)


def calc_sur_coord(seeds):
    around_seeds = np.array([[item + ind_formation] for item in seeds], dtype=np.short)[:, 0]
    return around_seeds


@numba.njit(nopython=True)
def check_coords_if_belong(array_3d, arrounds, r, s_coord, on_surface, last_ind):
    for cell_arrounds in arrounds:
        for point in cell_arrounds:
            if (point[2] - s_coord) ** 2 + (point[1] - s_coord) ** 2 + (point[0] - s_coord) ** 2 <= r ** 2:
                if not array_3d[point[0], point[1], point[2]]:
                    array_3d[point[0], point[1], point[2]] = True
                    on_surface[last_ind] = [point[0], point[1], point[2]]
                    last_ind += 1
    return last_ind


@numba.njit(nopython=True)
def go_around_bool(array_3d, arrounds):
    all_neighbours = []
    # trick to initialize an empty list with known type
    single_neighbours = [np.ubyte(x) for x in range(0)]
    for seed_arrounds in arrounds:
        for point in seed_arrounds:
            single_neighbours.append(array_3d[point[0], point[1], point[2]])
        all_neighbours.append(single_neighbours)
        single_neighbours = [np.ubyte(x) for x in range(0)]
    return np.array(all_neighbours, dtype=np.bool_)


# @numba.njit(nopython=True)
def fill_sphere_coords(cells_3d, s_coord, r):
    all_where_true = [np.short([x, x, x]) for x in range(0)]
    for z in range(1, cells_3d.shape[0] - 1):
        for y in range(1, cells_3d.shape[0] - 1):
            for x in range(1, cells_3d.shape[0] - 1):
                if (x - s_coord) ** 2 + (y - s_coord) ** 2 + (z - s_coord) ** 2 <= r ** 2:
                    cells_3d[z, y, x] = True
                    all_where_true.append(np.short([z, y, x]))
    return all_where_true


@numba.njit(nopython=True)
def aggregate(aggregated_ind, all_neigh_bool):
    # trick to initialize an empty list with known type
    where_blocks = [np.uint32(x) for x in range(0)]
    for index, item in enumerate(all_neigh_bool):
        for step in aggregated_ind:
            if np.sum(item[step]) == 7:
                where_blocks.append(np.uint32(index))
                break
    return np.array(where_blocks, dtype=np.uint32)


@numba.njit(nopython=True)
def aggregate_and_count(aggregated_ind, all_neigh_bool):
    # trick to initialize an empty list with known type
    block_counts = [np.uint32(x) for x in range(0)]
    for item in all_neigh_bool:
        curr_count = 0
        for step in aggregated_ind:
            if np.sum(item[step]) == 7:
                curr_count += 1
        block_counts.append(np.uint32(curr_count))
    return np.array(block_counts, dtype=np.uint32)


def calc_mean_neigh(real_radius):
    # n_cells_per_axis = 2 * step + 3
    # real_radius = (2 * step + 1) / 2
    # shift = step + 1

    step = math.ceil(real_radius - 1 / 2)
    n_cells_per_axis = 2 * step + 3
    shift = step + 1

    shape = (n_cells_per_axis, n_cells_per_axis, n_cells_per_axis)
    c3d = np.array(np.full(shape, False, dtype=bool))

    belong_to_sphere = fill_sphere_coords(c3d, shift, real_radius)
    flat_arrounds = calc_sur_coord(belong_to_sphere)
    del belong_to_sphere
    gc.collect()

    neighbours = go_around_bool(c3d, flat_arrounds)
    del flat_arrounds, c3d
    gc.collect()

    flat_arr_len = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)

    inside_ind = np.array(np.where(flat_arr_len == 6)[0])
    on_surface = np.delete(flat_arr_len, inside_ind)

    # unique_numbs = np.array(np.unique(on_surface, return_counts=True))
    # freq = np.array(unique_numbs[1] / np.sum(unique_numbs[1]))

    # return np.mean(on_surface), unique_numbs[0], freq
    return np.mean(on_surface)


if __name__ == "__main__":
    sphere = Sphere(user_input)

    for step in range(500):
        print(step)
        sphere.calc_mean_neigh()

    # sphere.plot_on_surface()


    # print("R NO_Block_mean 0 1 2 3 4 5 Block_mean 3 4 5")
    print("R 0 1 2 3 4 5 6 7 8")
    for key in sphere.stats:
        print(key, sphere.stats[key]["Block_counts"][0], sphere.stats[key]["Block_counts"][1], sphere.stats[key]["Block_counts"][2], sphere.stats[key]["Block_counts"][3], sphere.stats[key]["Block_counts"][4], sphere.stats[key]["Block_counts"][5], sphere.stats[key]["Block_counts"][6], sphere.stats[key]["Block_counts"][7], sphere.stats[key]["Block_counts"][8], sep=" ")
