import numba
import numpy as np
# from scipy.special import dtype

# Cache is True by default, but we can set it to False to force Numba to recompile the function on each call
_CACHE = False


@numba.njit(fastmath=True, cache=_CACHE)
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


@numba.njit(fastmath=True, cache=_CACHE)
def separate_in_interface(scale, arounds):
    out_int = [np.uint32(x) for x in range(0)]
    in_int = [np.uint32(x) for x in range(0)]
    blocked = [np.uint32(x) for x in range(0)]
    single_neighbours = [bool(x) for x in range(0)]
    for index, seed_arounds in enumerate(arounds):
        if scale[seed_arounds[-1][0], seed_arounds[-1][1], seed_arounds[-1][2]]:
            for point in seed_arounds[:-1]:
                single_neighbours.append(bool(scale[point[0], point[1], point[2]]))
            if 0 < sum(single_neighbours) < 4:
                in_int.append(np.uint32(index))
            else:
                blocked.append(np.uint32(index))
        else:
            out_int.append(np.uint32(index))
        single_neighbours = [bool(x) for x in range(0)]
    return np.array(in_int, dtype=np.uint32), np.array(blocked, dtype=np.uint32), np.array(out_int, dtype=np.uint32)


@numba.njit(fastmath=True, cache=_CACHE)
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


@numba.njit(fastmath=True, cache=_CACHE)
def go_around_int_and_summ(array_3d, arounds):
    all_neighbours = []
    # trick to initialize an empty list with known type
    single_neighbours = [np.ubyte(x) for x in range(0)]
    for seed_arounds in arounds:
        for point in seed_arounds:
            single_neighbours.append(array_3d[point[0], point[1], point[2]])
        all_neighbours.append(single_neighbours)
        single_neighbours = [np.ubyte(x) for x in range(0)]
    return np.array(all_neighbours, dtype=np.ubyte)


@numba.njit(fastmath=True, cache=_CACHE)
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


@numba.njit(fastmath=True, cache=_CACHE)
def check_at_coord_dissol(array_3d, coords):
    # trick to initialize an empty list with known type
    result_coords = [np.uint32(x) for x in range(0)]
    for coordinate in coords.transpose():
        result_coords.append(array_3d[coordinate[0], coordinate[1], coordinate[2]])
        # where_full.append(np.uint32(index))
    return np.array(result_coords, dtype=np.uint32)


@numba.njit(fastmath=True, cache=_CACHE)
def check_at_coord(array_3d, coordinates):
    # trick to initialize an empty list with known type
    result_coords = [np.bool_(x) for x in range(0)]
    for single_coordinate in coordinates:
        result_coords.append(array_3d[single_coordinate[0], single_coordinate[1], single_coordinate[2]])
    return np.array(result_coords, dtype=np.bool_)


@numba.njit(fastmath=True, cache=_CACHE)
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


@numba.njit(fastmath=True, cache=_CACHE)
def insert_counts(array_3d, points, threshold):
    for point in points.transpose():
        array_3d[point[0], point[1], point[2]] += threshold


@numba.njit(fastmath=True, cache=_CACHE)
def decrease_counts(array_3d, points):
    zero_positions = []
    for ind, point in enumerate(points.transpose()):
        if array_3d[point[0], point[1], point[2]] > 0:
            array_3d[point[0], point[1], point[2]] -= 1
        else:
            zero_positions.append(ind)
    return zero_positions


@numba.njit(fastmath=True, cache=_CACHE)
def just_decrease_counts(array_3d, points):
    for point in points.transpose():
        array_3d[point[0], point[1], point[2]] -= 1


@numba.njit(fastmath=True, cache=_CACHE)
def check_in_scale(scale, cells, dirs):
    # trick to initialize an empty list with known type
    out_scale = [np.uint32(x) for x in range(0)]
    for index, coordinate in enumerate(cells.transpose()):
        if not scale[coordinate[0], coordinate[1], coordinate[2]]:
            out_scale.append(np.uint32(index))
        else:
            dirs[:, index] *= -1
    return np.array(out_scale, dtype=np.uint32)


@numba.njit(fastmath=True, cache=_CACHE)
def check_in_scale_adj(scale, cells):
    # trick to initialize an empty list with known type
    out_scale = [np.uint32(x) for x in range(0)]
    in_scale = [np.uint32(x) for x in range(0)]
    for index, coordinate in enumerate(cells.transpose()):
        if not scale[coordinate[0], coordinate[1], coordinate[2]]:
            out_scale.append(np.uint32(index))
        else:
            in_scale.append(np.uint32(index))
    return np.array(out_scale, dtype=np.uint32), np.array(in_scale, dtype=np.uint32)


@numba.njit(fastmath=True, cache=_CACHE)
def check_in_scale_mp(scale, cells, dirs, working_range):
    # trick to initialize an empty list with known type
    out_scale = [np.uint32(x) for x in range(0)]
    for index, coordinate in enumerate(cells[:, working_range].transpose()):
        if not scale[coordinate[0], coordinate[1], coordinate[2]]:
            out_scale.append(np.uint32(index))
        else:
            dirs[:, working_range[index]] *= -1
    return np.array(out_scale, dtype=np.uint32)


@numba.njit(fastmath=True, cache=_CACHE)
def check_in_scale_mp_adj(scale, cells, working_range):
    # trick to initialize an empty list with known type
    out_scale = [np.uint32(x) for x in range(0)]
    in_scale = [np.uint32(x) for x in range(0)]
    for index, coordinate in enumerate(cells[:, working_range].transpose()):
        if not scale[coordinate[0], coordinate[1], coordinate[2]]:
            out_scale.append(np.uint32(index))
        else:
            in_scale.append(np.uint32(index))
    return np.array(out_scale, dtype=np.uint32), np.array(in_scale, dtype=np.uint32)


@numba.njit(fastmath=True, cache=_CACHE)
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


@numba.njit(fastmath=True, cache=_CACHE)
def aggregate(aggregated_ind, all_neigh_bool):
    # trick to initialize an empty list with known type
    where_blocks = [np.uint32(x) for x in range(0)]
    for index, item in enumerate(all_neigh_bool):
        for step in aggregated_ind:
            if np.sum(item[step]) == 7:
                where_blocks.append(np.uint32(index))
                break
    return np.array(where_blocks, dtype=np.uint32)


@numba.njit(fastmath=True, cache=_CACHE)
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


@numba.njit(fastmath=True, cache=_CACHE)
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


@numba.njit(fastmath=True, cache=_CACHE)
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


@numba.njit(fastmath=True, cache=_CACHE)
def diffuse_bulk_chunk(cells, dirs, start, end, p1, p2, p3, p4, p_r, cells_per_axis,
                      boundary_z_left=0, boundary_z_right=1, boundary_z_periodic=1):
    """
    Chopard-Droz diffusion for a contiguous chunk [start:end] of particles.
    Modifies cells and dirs in place. Returns indices (global) of particles that left the z-domain.

    Z-boundary is set by the default args above; change them here to choose:
      boundary_z_left:   0 = open (delete when z < 0),  1 = reflect (z=1, dirs[2]=1)
      boundary_z_right:  0 = open (delete when z > max),  1 = reflect (z=max-2, dirs[2]=-1)
      boundary_z_periodic: 0 = use left/right, 1 = periodic z (wrap, no deletion)
    """
    chunk_size = end - start
    out_buf = np.empty(chunk_size, dtype=np.int64)
    n_out = 0
    max_z = cells_per_axis - 1
    for i in range(start, end):
        r = np.random.random()
        d0 = dirs[0, i]
        d1 = dirs[1, i]
        d2 = dirs[2, i]
        if r <= p1:
            dirs[0, i], dirs[1, i], dirs[2, i] = d2, d0, d1
        elif r <= p2:
            dirs[0, i], dirs[1, i], dirs[2, i] = -d2, -d0, -d1
        elif r <= p3:
            dirs[0, i], dirs[1, i], dirs[2, i] = d1, d2, d0
        elif r <= p4:
            dirs[0, i], dirs[1, i], dirs[2, i] = -d1, -d2, -d0
        elif r <= p_r:
            dirs[0, i], dirs[1, i], dirs[2, i] = -d0, -d1, -d2

        cells[0, i] += dirs[0, i]
        cells[1, i] += dirs[1, i]
        cells[2, i] += dirs[2, i]

        # periodic x, y
        if cells[0, i] < 0:
            cells[0, i] += cells_per_axis
        elif cells[0, i] >= cells_per_axis:
            cells[0, i] -= cells_per_axis
        if cells[1, i] < 0:
            cells[1, i] += cells_per_axis
        elif cells[1, i] >= cells_per_axis:
            cells[1, i] -= cells_per_axis

        # z: open (delete), reflect, or periodic
        z = cells[2, i]
        if boundary_z_periodic == 1:
            z = ((z % cells_per_axis) + cells_per_axis) % cells_per_axis
            cells[2, i] = z
        else:
            if z < 0:
                if boundary_z_left == 1:
                    cells[2, i] = 1
                    dirs[2, i] = 1
                else:
                    out_buf[n_out] = i
                    n_out += 1
            elif z > max_z:
                if boundary_z_right == 1:
                    cells[2, i] = max_z - 1
                    dirs[2, i] = -1
                else:
                    out_buf[n_out] = i
                    n_out += 1
    return out_buf[:n_out]


@numba.njit(fastmath=True, cache=_CACHE)
def init_particles_rand(count, dirs, n, n2, max_per_cell, total_particles, packed_dirs, k_lo, seed):
    """Place total_particles randomly in 3D grid; cap at max_per_cell per cell. Fills count and dirs in place."""
    np.random.seed(seed)
    placed = 0
    while placed < total_particles:
        k = np.random.randint(k_lo, n)
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        idx = i + n * j + n2 * k
        if count[idx] < max_per_cell:
            c = count[idx]
            dirs[idx, c] = packed_dirs[np.random.randint(0, 6)]
            count[idx] = c + 1
            placed += 1


@numba.njit(fastmath=True, cache=_CACHE)
def init_particles_exact(count, dirs, n, n2, max_per_cell, n_per, k_lo, packed_dirs, seed):
    """Place n_per particles per z-slice (k_lo..n-1), random 2D position without replacement per slice. Fills count and dirs in place."""
    np.random.seed(seed)
    nn = n * n
    for k in range(k_lo, n):
        # Fisher–Yates shuffle to get n_per distinct 2D indices
        arr = np.arange(nn)
        for i in range(nn - 1, nn - n_per - 1, -1):
            j = np.random.randint(0, i + 1)
            arr[i], arr[j] = arr[j], arr[i]
        for p in range(n_per):
            idx_2d = arr[nn - 1 - p]
            i = idx_2d % n
            j = idx_2d // n
            idx = i + n * j + n2 * k
            if count[idx] < max_per_cell:
                c = count[idx]
                dirs[idx, c] = packed_dirs[np.random.randint(0, 6)]
                count[idx] = c + 1


@numba.njit(fastmath=True, cache=_CACHE)
def fill_first_page_kernel(count, dirs, n, n2, max_per_cell, num_to_add, dir_packed, seed):
    """Add num_to_add particles on x=0 plane (zy plane); random (j,k), idx = n*j + n2*k. In place."""
    np.random.seed(seed)
    placed = 0
    while placed < num_to_add:
        j = np.random.randint(0, n)
        k = np.random.randint(0, n)
        idx = n * j + n2 * k
        if count[idx] < max_per_cell:
            c = count[idx]
            dirs[idx, c] = dir_packed
            count[idx] = c + 1
            placed += 1
