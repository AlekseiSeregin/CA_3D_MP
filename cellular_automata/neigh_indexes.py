from configuration import Config
import numpy as np


def generate_neigh_indexes_flat():
    # Neighbour offsets: convention index 0=x, 1=y, 2=z. For zy-plane we fix x=0 (zero only index 0).
    size = 3 + (Config.NEIGH_RANGE - 1) * 2
    neigh_shape = (size, size, 3)
    temp = np.ones(neigh_shape, dtype=int)
    temp[:, :, 0] = 0  # x=0 only (zy-plane); do not zero index 2 (z)

    # Zero only the 6 axis-aligned neighbours (exclude center 0,0,0 so it stays in the result)
    flat_ind = np.array(ind_decompose_flat_z[:-1])
    flat_ind = flat_ind.transpose()
    flat_ind[0] += Config.NEIGH_RANGE  # x
    flat_ind[1] += Config.NEIGH_RANGE  # y
    flat_ind[2] += 1                   # z (index 2 = z, not x)

    temp[flat_ind[0], flat_ind[1], flat_ind[2]] = 0

    coord = np.array(np.nonzero(temp))
    coord[0] -= Config.NEIGH_RANGE
    coord[1] -= Config.NEIGH_RANGE
    coord[2] -= 1
    coord = coord.transpose()
    # coord rows are (di, dj, axis); for zy-plane we want (0, dy, dz) = (0, di, dj), unique (di,dj)
    coord_2d = np.unique(coord[:, :2], axis=0)
    zy_plane = np.column_stack([np.zeros(coord_2d.shape[0], dtype=np.byte), coord_2d[:, 0], coord_2d[:, 1]])
    # First 9 = zy-plane (nucleation); last 2 = x±1 (growth only)
    growth_x = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.byte)
    coord = np.concatenate((zy_plane, growth_x))

    return np.array(coord, dtype=np.byte)


ind_decompose_flat_z = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 0]], dtype=np.byte)

ind_decompose_no_flat = np.array(
            [[1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],
             [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1],
             [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],
             [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=np.byte)

# Single 26-neighbour offsets for dissolution: indices 0–5 = face (flat), 6–25 = non-flat. Never modified.
OFFSETS_26 = np.vstack((ind_decompose_flat_z[:6], ind_decompose_no_flat))

ind_formation = generate_neigh_indexes_flat()


def calc_sur_ind_decompose_flat_with_zero(seeds):
    seeds = seeds.transpose()
    # generating a neighbouring coordinates for each seed (including the position of the seed itself)
    around_seeds = np.array([[item + ind_decompose_flat_z] for item in seeds], dtype=np.short)[:, 0]
    # applying periodic boundary conditions
    around_seeds[around_seeds == Config.N_CELLS_PER_AXIS] = 0
    around_seeds[around_seeds == -1] = Config.N_CELLS_PER_AXIS - 1
    return around_seeds


def calc_sur_ind_decompose_no_flat(seeds):
    seeds = seeds.transpose()
    # generating a neighbouring coordinates for each seed (including the position of the seed itself)
    around_seeds = np.array([[item + ind_decompose_no_flat] for item in seeds], dtype=np.short)[:, 0]
    # applying periodic boundary conditions
    around_seeds[around_seeds == Config.N_CELLS_PER_AXIS] = 0
    around_seeds[around_seeds == -1] = Config.N_CELLS_PER_AXIS - 1
    return around_seeds


def calc_sur_ind_formation(seeds, dummy_ind):
    # generating a neighbouring coordinates for each seed (including the position of the seed itself)
    around_seeds = np.array([[item + ind_formation] for item in seeds], dtype=np.short)[:, 0]
    # applying periodic boundary conditions
    if seeds[0, 2] < Config.NEIGH_RANGE:
        indexes = np.where(around_seeds[:, :, 2] < 0)
        around_seeds[indexes[0], indexes[1], 2] = dummy_ind
    for shift in range(Config.NEIGH_RANGE):
        indexes = np.where(around_seeds[:, :, 0:2] == Config.N_CELLS_PER_AXIS + shift)
        around_seeds[indexes[0], indexes[1], indexes[2]] = shift
        indexes = np.where(around_seeds[:, :, 0:2] == - shift - 1)
        around_seeds[indexes[0], indexes[1], indexes[2]] = Config.N_CELLS_PER_AXIS - shift - 1
    return around_seeds
