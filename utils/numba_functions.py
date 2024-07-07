import numba
import numpy as np

# @numba.njit(nopython=True)


@numba.njit(nopython=True, fastmath=True)
def go_around_bool(array_3d, arounds):
    all_neighbours = []
    # trick to initialize an empty list with known type
    single_neighbours = [np.ubyte(x) for x in range(0)]
    for seed_arounds in arounds:
        for point in seed_arounds:
            single_neighbours.append(array_3d[point[0], point[1], point[2]])
        all_neighbours.append(single_neighbours)
        single_neighbours = [np.ubyte(x) for x in range(0)]
    return np.array(all_neighbours, dtype=np.bool_)


@numba.njit(nopython=True, fastmath=True)
def go_around_int(array_3d, arounds):
    all_neighbours = []
    # trick to initialize an empty list with known type
    single_neighbours = [np.ubyte(x) for x in range(0)]
    for seed_arounds in arounds:
        for point in seed_arounds:
            single_neighbours.append(array_3d[point[0], point[1], point[2]])
        all_neighbours.append(single_neighbours)
        single_neighbours = [np.ubyte(x) for x in range(0)]
    return np.array(all_neighbours, dtype=np.ubyte)


@numba.njit(nopython=True, fastmath=True)
def go_around_bool_dissol(array_3d, arounds):
    all_neigh = []
    # trick to initialize an empty list with known type
    single_neigh = [np.bool_(x) for x in range(0)]
    for seed_arounds in arounds:
        for point in seed_arounds:
            single_neigh.append(array_3d[point[0], point[1], point[2]])
        all_neigh.append(single_neigh)
        single_neigh = [np.bool_(x) for x in range(0)]
    return np.array(all_neigh, dtype=np.bool_)


@numba.njit(nopython=True, fastmath=True)
def check_at_coord_dissol(array_3d, coords):
    # trick to initialize an empty list with known type
    result_coords = [np.uint32(x) for x in range(0)]
    for coordinate in coords.transpose():
        result_coords.append(array_3d[coordinate[0], coordinate[1], coordinate[2]])
        # where_full.append(np.uint32(index))
    return np.array(result_coords, dtype=np.uint32)


@numba.njit(nopython=True, fastmath=True)
def check_at_coord(array_3d, coordinates):
    # trick to initialize an empty list with known type
    result_coords = [np.bool_(x) for x in range(0)]
    for single_coordinate in coordinates:
        result_coords.append(array_3d[single_coordinate[0], single_coordinate[1], single_coordinate[2]])
    return np.array(result_coords, dtype=np.bool_)


@numba.njit(nopython=True, fastmath=True)
def check_at_coord_new(array_3d, coordinates):
    # trick to initialize an empty list with known type
    result_ind = [np.uint32(x) for x in range(0)]
    counts = [np.ubyte(x) for x in range(0)]
    for index, coord in enumerate(coordinates):
        array_val = array_3d[coord[0], coord[1], coord[2]]
        if array_val:
            result_ind.append(np.uint32(index))
            counts.append(np.ubyte(array_val))
    return np.array(result_ind, dtype=np.uint32), np.array(counts, dtype=np.ubyte)


@numba.njit(nopython=True, fastmath=True)
def insert_counts(array_3d, points):
    for point in points.transpose():
        array_3d[point[0], point[1], point[2]] += 1


@numba.njit(nopython=True, fastmath=True)
def decrease_counts(array_3d, points):
    zero_positions = []
    for ind, point in enumerate(points.transpose()):
        if array_3d[point[0], point[1], point[2]] > 0:
            array_3d[point[0], point[1], point[2]] -= 1
        else:
            zero_positions.append(ind)
    return zero_positions


@numba.njit(nopython=True, fastmath=True)
def just_decrease_counts(array_3d, points):
    for point in points.transpose():
        array_3d[point[0], point[1], point[2]] -= 1


@numba.njit(nopython=True, fastmath=True)
def check_in_scale(scale, cells, dirs):
    # trick to initialize an empty list with known type
    out_scale = [np.uint32(x) for x in range(0)]
    for index, coordinate in enumerate(cells.transpose()):
        if not scale[coordinate[0], coordinate[1], coordinate[2]]:
            out_scale.append(np.uint32(index))
        else:
            dirs[:, index] *= -1
    return np.array(out_scale, dtype=np.uint32)


@numba.njit(nopython=True, fastmath=True)
def separate_in_gb(bool_arr):
    # trick to initialize an empty list with known type
    out_gb = [np.uint32(x) for x in range(0)]
    in_gb = [np.uint32(x) for x in range(0)]
    for index, bool_item in enumerate(bool_arr):
        if bool_item:
            in_gb.append(np.uint32(index))
        else:
            out_gb.append(np.uint32(index))
    return np.array(in_gb, dtype=np.uint32), np.array(out_gb, dtype=np.uint32)


@numba.njit(nopython=True, fastmath=True)
def aggregate(aggregated_ind, all_neigh_bool):
    # trick to initialize an empty list with known type
    where_blocks = [np.uint32(x) for x in range(0)]
    for index, item in enumerate(all_neigh_bool):
        for step in aggregated_ind:
            if np.sum(item[step]) == 7:
                where_blocks.append(np.uint32(index))
                break
    return np.array(where_blocks, dtype=np.uint32)


@numba.njit(nopython=True, fastmath=True)
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


@numba.njit(nopython=True, fastmath=True)
def diff_single(directions, probs, random_numbs):
    for index, direction in enumerate(directions.transpose()):
        rand_numb = random_numbs.random()
        if rand_numb > probs[4]:
            new_direction = [direction[0], direction[1], direction[2]]
        elif probs[3] < rand_numb <= probs[4]:
            new_direction = [np.byte(direction[0] * -1), np.byte(direction[1] * -1), np.byte(direction[2] * -1)]
        elif rand_numb <= probs[0]:
            new_direction = [direction[2], direction[0], direction[1]]
        elif probs[0] < rand_numb <= probs[1]:
            new_direction = [np.byte(direction[2] * -1), np.byte(direction[0] * -1), np.byte(direction[1] * -1)]
        elif probs[1] < rand_numb <= probs[2]:
            new_direction = [direction[1], direction[2], direction[0]]
        else:
            new_direction = [np.byte(direction[1] * -1), np.byte(direction[2] * -1), np.byte(direction[0] * -1)]
        directions[:, index] = new_direction


@numba.njit(nopython=True, fastmath=True)
def complete_diff_step(cells, directions, probs, random_numbs):
    for index, direction in enumerate(directions.transpose()):
        rand_numb = random_numbs.random()
        if rand_numb > probs[4]:
            new_direction = [direction[0], direction[1], direction[2]]
        elif probs[3] < rand_numb <= probs[4]:
            new_direction = [np.byte(direction[0] * -1), np.byte(direction[1] * -1), np.byte(direction[2] * -1)]
        elif rand_numb <= probs[0]:
            new_direction = [direction[2], direction[0], direction[1]]
        elif probs[0] < rand_numb <= probs[1]:
            new_direction = [np.byte(direction[2] * -1), np.byte(direction[0] * -1), np.byte(direction[1] * -1)]
        elif probs[1] < rand_numb <= probs[2]:
            new_direction = [direction[1], direction[2], direction[0]]
        else:
            new_direction = [np.byte(direction[1] * -1), np.byte(direction[2] * -1), np.byte(direction[0] * -1)]

        directions[:, index] = new_direction

        new_direction = [np.short(direction[0]), np.short(direction[1]), np.short(direction[2])]
        coord = cells[:, index]
        new_coord = [a + b for a, b in zip(new_direction, coord)]
        cells[:, index] = new_coord

        if cells[2, index] < 0:
            cells[2, index] = 1
            directions[2, index] = 1

        elif cells[0, index] == -1:
            cells[0, index] = 500

        elif cells[0, index] == 501:
            cells[0, index] = 0

        elif cells[1, index] == -1:
            cells[1, index] = 500

        elif cells[1, index] == 501:
            cells[1, index] = 0

        elif cells[2, index] == 501:
            cells[2, index] = 499
            directions[2, index] = -1



