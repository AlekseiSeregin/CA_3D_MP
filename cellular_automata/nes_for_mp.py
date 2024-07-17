class SharedMetaData:
    def __init__(self, shm_name, shape, dtype):
        self.name = shm_name
        self.shape = shape
        self.dtype = dtype


class PRanges:
    def __init__(self, p1_range, p2_range, p3_range, p4_range, p_r_range):
        self.p1_range = p1_range
        self.p2_range = p2_range
        self.p3_range = p3_range
        self.p4_range = p4_range
        self.p_r_range = p_r_range


def worker(args):
    callback = args[-1]
    args = args[:-1]
    result = callback(*args)
    return result

# def generate_neigh_indexes_flat():
#     size = 3 + (Config.NEIGH_RANGE - 1) * 2
#     neigh_shape = (size, size, 3)
#     temp = np.ones(neigh_shape, dtype=int)
#     temp[:, :, 0] = 0
#     temp[:, :, 2] = 0
#
#     flat_ind = np.array(ind_decompose_flat_z)
#     flat_ind = flat_ind.transpose()
#     flat_ind[0] += Config.NEIGH_RANGE
#     flat_ind[1] += Config.NEIGH_RANGE
#     flat_ind[2] += 1
#
#     temp[flat_ind[0], flat_ind[1], flat_ind[2]] = 0
#
#     coord = np.array(np.nonzero(temp))
#     coord[0] -= Config.NEIGH_RANGE
#     coord[1] -= Config.NEIGH_RANGE
#     coord[2] -= 1
#     coord = coord.transpose()
#
#     coord = np.concatenate((ind_decompose_flat_z, coord))
#     additional = coord[[2, 5]]
#     coord = np.delete(coord, [2, 5], axis=0)
#     coord = np.concatenate((coord, additional))
#
#     return np.array(coord, dtype=np.byte)
#
#
# ind_decompose_flat_z = np.array(
#             [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 0]], dtype=np.byte)
#
# ind_decompose_no_flat = np.array(
#             [[1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],
#              [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1],
#              [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],
#              [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=np.byte)
#
# ind_formation = generate_neigh_indexes_flat()
#
#
# def calc_sur_ind_decompose_flat_with_zero(seeds):
#     seeds = seeds.transpose()
#     # generating a neighbouring coordinates for each seed (including the position of the seed itself)
#     around_seeds = np.array([[item + ind_decompose_flat_z] for item in seeds], dtype=np.short)[:, 0]
#     # applying periodic boundary conditions
#     around_seeds[around_seeds == Config.N_CELLS_PER_AXIS] = 0
#     around_seeds[around_seeds == -1] = Config.N_CELLS_PER_AXIS - 1
#     return around_seeds
#
#
# def calc_sur_ind_decompose_no_flat(seeds):
#     seeds = seeds.transpose()
#     # generating a neighbouring coordinates for each seed (including the position of the seed itself)
#     around_seeds = np.array([[item + ind_decompose_no_flat] for item in seeds], dtype=np.short)[:, 0]
#     # applying periodic boundary conditions
#     around_seeds[around_seeds == Config.N_CELLS_PER_AXIS] = 0
#     around_seeds[around_seeds == -1] = Config.N_CELLS_PER_AXIS - 1
#     return around_seeds
#
#
# def calc_sur_ind_formation(seeds, dummy_ind):
#     # generating a neighbouring coordinates for each seed (including the position of the seed itself)
#     around_seeds = np.array([[item + ind_formation] for item in seeds], dtype=np.short)[:, 0]
#     # applying periodic boundary conditions
#     if seeds[0, 2] < Config.NEIGH_RANGE:
#         indexes = np.where(around_seeds[:, :, 2] < 0)
#         around_seeds[indexes[0], indexes[1], 2] = dummy_ind
#     for shift in range(Config.NEIGH_RANGE):
#         indexes = np.where(around_seeds[:, :, 0:2] == Config.N_CELLS_PER_AXIS + shift)
#         around_seeds[indexes[0], indexes[1], indexes[2]] = shift
#         indexes = np.where(around_seeds[:, :, 0:2] == - shift - 1)
#         around_seeds[indexes[0], indexes[1], indexes[2]] = Config.N_CELLS_PER_AXIS - shift - 1
#     return around_seeds
#
#
# def fix_full_cells(array_3d, full_array_3d, new_precip):
#     current_precip = np.array(array_3d[new_precip[0], new_precip[1], new_precip[2]], dtype=np.ubyte)
#     indexes = np.where(current_precip == 3)[0]
#     full_precip = new_precip[:, indexes]
#     full_array_3d[full_precip[0], full_precip[1], full_precip[2]] = True
#
#
# def go_around_mult_oxid_n_also_partial_neigh_aip_MP(array_3d, around_coords):
#     return np.sum(go_around_int(array_3d, around_coords), axis=1)
#
#
# def precip_step_standard_MP(product_x_nzs_mdata, shm_mdata_product, shm_mdata_full_product, shm_mdata_product_init,
#                             shm_mdata_active, shm_mdata_oxidant, plane_indexes, fetch_indexes, nucleation_probabilities,
#                             callback):
#     shm_o = shared_memory.SharedMemory(name=shm_mdata_oxidant.name)
#     oxidant = np.ndarray(shm_mdata_oxidant.shape, dtype=shm_mdata_oxidant.dtype, buffer=shm_o.buf)
#
#     shm_p_FULL = shared_memory.SharedMemory(name=shm_mdata_full_product.name)
#     full_3d = np.ndarray(shm_mdata_full_product.shape, dtype=shm_mdata_full_product.dtype, buffer=shm_p_FULL.buf)
#
#     for fetch_ind in fetch_indexes:
#         plane_indexes = np.array(plane_indexes)
#         # oxidant_cells = oxidant[fetch_ind[0], fetch_ind[1], plane_indexes]
#         oxidant_cells1 = oxidant[fetch_ind[0][:, np.newaxis], fetch_ind[1][:, np.newaxis], plane_indexes]
#         nonzero_indices = np.nonzero(oxidant_cells1)
#         oxidant_cells = fetch_ind[:, nonzero_indices[0]]
#         # nz = np.array(nonzero_indices[1])
#         # oxidant_cells = fetch_ind[:, oxidant_cells2]
#
#         if len(oxidant_cells[0]) != 0:
#             # n_plane_indexes = plane_indexes[np.array(nonzero_indices[1])]
#             # oxidant_cells = np.vstack((oxidant_cells, np.full(len(oxidant_cells[0]), plane_indexes)))
#             oxidant_cells = np.vstack((oxidant_cells, plane_indexes[np.array(nonzero_indices[1])]))
#
#             oxidant_cells = np.array(oxidant_cells, dtype=np.short).transpose()
#
#             exists = check_at_coord(full_3d, oxidant_cells)  # precip on place of oxidant!
#             temp_ind = np.where(exists)[0]
#
#             oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)
#
#             if len(oxidant_cells) > 0:
#                 # activate if microstructure ___________________________________________________________
#                 # in_gb = [self.microstructure[point[0], point[1], point[2]] for point in oxidant_cells]
#                 # temp_ind = np.where(in_gb)[0]
#                 # oxidant_cells = oxidant_cells[temp_ind]
#                 # ______________________________________________________________________________________
#                 callback(oxidant_cells, oxidant, full_3d, shm_mdata_product, shm_mdata_active, shm_mdata_product_init,
#                          nucleation_probabilities, product_x_nzs_mdata)
#
#     shm_o.close()
#     shm_p_FULL.close()
#
#
# def ci_single_MP(seeds, oxidant, full_3d, shm_mdata_product, shm_mdata_active, shm_mdata_product_init,
#                  nucleation_probabilities,
#                  product_x_nzs_mdata):
#     shm_p = shared_memory.SharedMemory(name=shm_mdata_product.name)
#     product = np.ndarray(shm_mdata_product.shape, dtype=shm_mdata_product.dtype, buffer=shm_p.buf)
#
#     shm_a = shared_memory.SharedMemory(name=shm_mdata_active.name)
#     active = np.ndarray(shm_mdata_active.shape, dtype=shm_mdata_active.dtype, buffer=shm_a.buf)
#
#     shm_product_init = shared_memory.SharedMemory(name=shm_mdata_product_init.name)
#     product_init = np.ndarray(shm_mdata_product_init.shape, dtype=shm_mdata_product_init.dtype,
#                               buffer=shm_product_init.buf)
#
#     shm_product_x_nzs = shared_memory.SharedMemory(name=product_x_nzs_mdata.name)
#     product_x_nzs = np.ndarray(product_x_nzs_mdata.shape, dtype=product_x_nzs_mdata.dtype,
#                                buffer=shm_product_x_nzs.buf)
#
#     all_arounds = calc_sur_ind_formation(seeds, active.shape[2] - 1)
#
#     neighbours = go_around_bool(active, all_arounds[:, :-2])
#     arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.short)
#
#     temp_ind = np.where(arr_len_out > 0)[0]
#
#     if len(temp_ind) > 0:
#         seeds = seeds[temp_ind]
#         neighbours = neighbours[temp_ind]
#         all_arounds = all_arounds[temp_ind]
#         flat_arounds = np.concatenate((all_arounds[:, 0:5], all_arounds[:, -2:]), axis=1)
#         arr_len_in_flat = go_around_mult_oxid_n_also_partial_neigh_aip_MP(product_init, flat_arounds)
#         homogeneous_ind = np.where(arr_len_in_flat == 0)[0]
#         needed_prob = nucleation_probabilities.get_probabilities(arr_len_in_flat, seeds[0][2])
#         needed_prob[homogeneous_ind] = nucleation_probabilities.nucl_prob.values_pp[seeds[0][2]]  # seeds[0][2] - current plane index
#         randomise = np.array(np.random.random_sample(arr_len_in_flat.size), dtype=np.float64)
#         temp_ind = np.where(randomise < needed_prob)[0]
#
#         if len(temp_ind) > 0:
#             seeds = seeds[temp_ind]
#             neighbours = neighbours[temp_ind]
#             all_arounds = all_arounds[temp_ind]
#
#             out_to_del = [np.array(np.nonzero(item)[0]) for item in neighbours]
#             to_del = [np.random.choice(item, 1, replace=False) for item in out_to_del]
#             coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
#                              dtype=np.short)
#
#             coord = np.reshape(coord, (len(coord) * 1, 3))
#             coord = coord.transpose()
#             seeds = seeds.transpose()
#
#             active[coord[0], coord[1], coord[2]] -= 1
#             oxidant[seeds[0], seeds[1], seeds[2]] -= 1
#
#             # self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
#             product[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!
#
#             # self.objs[self.case]["product"].fix_full_cells(coord)  # precip on place of active!
#             # self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!
#
#             fix_full_cells(product, full_3d, seeds)
#
#             # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
#             # dissolution function
#             product_x_nzs[seeds[2][0]] = True
#
#     shm_p.close()
#     shm_a.close()
#     shm_product_init.close()
#     shm_product_x_nzs.close()
#
#
# def dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL_MP(shm_mdata, chunk_range, comb_ind, aggregated_ind,
#                                                       dissolution_probabilities):
#     to_dissolve = np.array([[], [], []], dtype=np.short)
#     shm = shared_memory.SharedMemory(name=shm_mdata.name)
#     array_3D = np.ndarray(shm_mdata.shape, dtype=shm_mdata.dtype, buffer=shm.buf)
#     to_dissol_pn_buffer = np.array([[], [], []], dtype=np.short)
#
#     nz_ind = np.array(np.nonzero(array_3D[chunk_range[0]:chunk_range[1], :, comb_ind]))
#     nz_ind[0] += chunk_range[0]
#     coord_buffer = nz_ind
#
#     new_data = comb_ind[nz_ind[2]]
#     coord_buffer[2, :] = new_data
#
#     if len(coord_buffer[0]) > 0:
#         flat_arounds = calc_sur_ind_decompose_flat_with_zero(coord_buffer)
#
#         all_neigh = go_around_int(array_3D, flat_arounds)
#         all_neigh[:, 6] -= 1
#
#         all_neigh_block = np.array([])
#         all_neigh_no_block = np.array([])
#         numb_in_prod_block = np.array([], dtype=int)
#         numb_in_prod_no_block = np.array([], dtype=int)
#
#         where_not_null = np.unique(np.where(all_neigh[:, :6] > 0)[0])
#         to_dissol_no_neigh = np.array(np.delete(coord_buffer, where_not_null, axis=1), dtype=np.short)
#         coord_buffer = coord_buffer[:, where_not_null]
#
#         if len(coord_buffer[0]) > 0:
#             all_neigh = all_neigh[where_not_null]
#             numb_in_prod = all_neigh[:, -1]
#
#             all_neigh_bool = np.array(all_neigh[:, :6], dtype=bool)
#
#             arr_len_flat = np.sum(all_neigh_bool, axis=1)
#
#             index_outside = np.where((arr_len_flat < 6))[0]
#             coord_buffer = coord_buffer[:, index_outside]
#
#             all_neigh_bool = all_neigh_bool[index_outside]
#             arr_len_flat = arr_len_flat[index_outside]
#             numb_in_prod = numb_in_prod[index_outside]
#
#             non_flat_arounds = calc_sur_ind_decompose_no_flat(coord_buffer)
#             non_flat_neigh = go_around_bool(array_3D, non_flat_arounds)
#             all_neigh_bool = np.concatenate((all_neigh_bool, non_flat_neigh), axis=1)
#             ind_where_blocks = aggregate(aggregated_ind, all_neigh_bool)
#
#             if len(ind_where_blocks) > 0:
#                 to_dissol_pn_buffer = np.array(np.delete(coord_buffer, ind_where_blocks, axis=1), dtype=np.short)
#
#                 all_neigh_no_block = np.delete(arr_len_flat, ind_where_blocks)
#                 numb_in_prod_no_block = np.delete(numb_in_prod, ind_where_blocks, axis=0)
#                 coord_buffer = coord_buffer[:, ind_where_blocks]
#                 all_neigh_block = arr_len_flat[ind_where_blocks]
#
#                 numb_in_prod_block = numb_in_prod[ind_where_blocks]
#             else:
#                 to_dissol_pn_buffer = coord_buffer
#                 all_neigh_no_block = arr_len_flat
#                 numb_in_prod_no_block = numb_in_prod
#
#                 coord_buffer = np.array([[], [], []], dtype=np.ushort)
#                 all_neigh_block = np.array([])
#                 numb_in_prod_block = np.array([], dtype=int)
#
#         to_dissolve_no_block = to_dissol_pn_buffer
#         probs_no_block = dissolution_probabilities.get_probabilities(all_neigh_no_block, to_dissolve_no_block[2])
#
#         non_z_ind = np.where(numb_in_prod_no_block != 0)[0]
#         repeated_coords = np.repeat(to_dissolve_no_block[:, non_z_ind], numb_in_prod_no_block[non_z_ind], axis=1)
#         repeated_probs = np.repeat(probs_no_block[non_z_ind], numb_in_prod_no_block[non_z_ind])
#         to_dissolve_no_block = np.concatenate((to_dissolve_no_block, repeated_coords), axis=1)
#         probs_no_block = np.concatenate((probs_no_block, repeated_probs))
#         randomise = np.random.random_sample(len(to_dissolve_no_block[0]))
#         temp_ind = np.where(randomise < probs_no_block)[0]
#         to_dissolve_no_block = to_dissolve_no_block[:, temp_ind]
#
#         to_dissolve_block = coord_buffer
#         probs_block = dissolution_probabilities.get_probabilities_block(all_neigh_block, to_dissolve_block[2])
#
#         non_z_ind = np.where(numb_in_prod_block != 0)[0]
#         repeated_coords = np.repeat(to_dissolve_block[:, non_z_ind], numb_in_prod_block[non_z_ind], axis=1)
#         repeated_probs = np.repeat(probs_block[non_z_ind], numb_in_prod_block[non_z_ind])
#         to_dissolve_block = np.concatenate((to_dissolve_block, repeated_coords), axis=1)
#         probs_block = np.concatenate((probs_block, repeated_probs))
#         randomise = np.random.random_sample(len(to_dissolve_block[0]))
#         temp_ind = np.where(randomise < probs_block)[0]
#         to_dissolve_block = to_dissolve_block[:, temp_ind]
#
#         probs_no_neigh = dissolution_probabilities.dissol_prob.values_pp[to_dissol_no_neigh[2]]
#         randomise = np.random.random_sample(len(to_dissol_no_neigh[0]))
#         temp_ind = np.where(randomise < probs_no_neigh)[0]
#         to_dissol_no_neigh = to_dissol_no_neigh[:, temp_ind]
#
#         to_dissolve = np.concatenate((to_dissolve_no_block, to_dissol_no_neigh, to_dissolve_block), axis=1)
#
#     shm.close()
#     return to_dissolve
#
# def worker(input_queue, output_queue):
#     while True:
#         args = input_queue.get()
#         if args is None:
#             break
#         callback = args[-1]
#         args = args[:-1]
#         result = callback(*args)
#         output_queue.put(result)
