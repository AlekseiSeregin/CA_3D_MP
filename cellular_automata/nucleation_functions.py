from utils.numba_functions import *
from multiprocessing import shared_memory
from .neigh_indexes import *


# def precip_step_standard_MP(product_x_nzs_mdata, shm_mdata_product, shm_mdata_full_product, shm_mdata_product_init,
#                             shm_mdata_active, shm_mdata_oxidant, plane_indexes, fetch_indexes, nucleation_probabilities,
#                             callback):
#     shm_o = shared_memory.SharedMemory(name=shm_mdata_oxidant.name)
#     oxidant = np.ndarray(shm_mdata_oxidant.shape, dtype=shm_mdata_oxidant.dtype, buffer=shm_o.buf)
#     shm_p_FULL = shared_memory.SharedMemory(name=shm_mdata_full_product.name)
#     full_3d = np.ndarray(shm_mdata_full_product.shape, dtype=shm_mdata_full_product.dtype, buffer=shm_p_FULL.buf)
#
#     for fetch_ind in fetch_indexes:
#         plane_indexes = np.array(plane_indexes)
#         nonzero_indices = np.nonzero(oxidant[fetch_ind[0][:, np.newaxis], fetch_ind[1][:, np.newaxis], plane_indexes])
#         oxidant_cells = fetch_ind[:, nonzero_indices[0]]
#
#         if len(oxidant_cells[0]) != 0:
#             oxidant_cells = np.vstack((oxidant_cells, plane_indexes[np.array(nonzero_indices[1])]))
#             oxidant_cells = np.array(oxidant_cells, dtype=np.short).transpose()
#             exists = check_at_coord(full_3d, oxidant_cells)  # precip on place of oxidant!
#             temp_ind = np.where(exists)[0]
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
#     shm_o.close()
#     shm_p_FULL.close()

def precip_step_standard(cur_case, plane_indexes, fetch_indexes, callback):

    shm_o = shared_memory.SharedMemory(name=cur_case.oxidant_c3d_shm_mdata.name)
    oxidant = np.ndarray(cur_case.oxidant_c3d_shm_mdata.shape, dtype=cur_case.oxidant_c3d_shm_mdata.dtype, buffer=shm_o.buf)

    shm_p_FULL = shared_memory.SharedMemory(name=cur_case.full_shm_mdata.name)
    full_3d = np.ndarray(cur_case.full_shm_mdata.shape, dtype=cur_case.full_shm_mdata.dtype, buffer=shm_p_FULL.buf)

    for fetch_ind in fetch_indexes:
        plane_indexes = np.array(plane_indexes)
        nonzero_indices = np.nonzero(oxidant[fetch_ind[0][:, np.newaxis], fetch_ind[1][:, np.newaxis], plane_indexes])
        oxidant_cells = fetch_ind[:, nonzero_indices[0]]

        if len(oxidant_cells[0]) != 0:
            oxidant_cells = np.vstack((oxidant_cells, plane_indexes[np.array(nonzero_indices[1])]))
            oxidant_cells = np.array(oxidant_cells, dtype=np.short).transpose()
            exists = check_at_coord(full_3d, oxidant_cells)  # precip on place of oxidant!
            temp_ind = np.where(exists)[0]
            oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

            if len(oxidant_cells) > 0:
                # activate if microstructure ___________________________________________________________
                # in_gb = [self.microstructure[point[0], point[1], point[2]] for point in oxidant_cells]
                # temp_ind = np.where(in_gb)[0]
                # oxidant_cells = oxidant_cells[temp_ind]
                # ______________________________________________________________________________________
                callback(cur_case, oxidant_cells, oxidant, full_3d)
    shm_o.close()
    shm_p_FULL.close()

# def ci_single_MP(seeds, oxidant, full_3d, shm_mdata_product, shm_mdata_active, shm_mdata_product_init,
#                  nucleation_probabilities,
#                  product_x_nzs_mdata):
#     shm_p = shared_memory.SharedMemory(name=shm_mdata_product.name)
#     product = np.ndarray(shm_mdata_product.shape, dtype=shm_mdata_product.dtype, buffer=shm_p.buf)
#     shm_a = shared_memory.SharedMemory(name=shm_mdata_active.name)
#     active = np.ndarray(shm_mdata_active.shape, dtype=shm_mdata_active.dtype, buffer=shm_a.buf)
#     shm_product_init = shared_memory.SharedMemory(name=shm_mdata_product_init.name)
#     product_init = np.ndarray(shm_mdata_product_init.shape, dtype=shm_mdata_product_init.dtype,
#                               buffer=shm_product_init.buf)
#     shm_product_x_nzs = shared_memory.SharedMemory(name=product_x_nzs_mdata.name)
#     product_x_nzs = np.ndarray(product_x_nzs_mdata.shape, dtype=product_x_nzs_mdata.dtype,
#                                buffer=shm_product_x_nzs.buf)
#
#     all_arounds = calc_sur_ind_formation(seeds, active.shape[2] - 1)
#     neighbours = go_around_bool(active, all_arounds[:, :-2])
#     arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.short)
#     temp_ind = np.where(arr_len_out > 0)[0]
#
#     if len(temp_ind) > 0:
#         seeds = seeds[temp_ind]
#         neighbours = neighbours[temp_ind]
#         all_arounds = all_arounds[temp_ind]
#
#         flat_arounds = np.concatenate((all_arounds[:, 0:5], all_arounds[:, -2:]), axis=1)
#
#         arr_len_in_flat = go_around_mult_oxid_n_also_partial_neigh_aip_MP(product_init, flat_arounds)
#
#         homogeneous_ind = np.where(arr_len_in_flat == 0)[0]
#         # print(homogeneous_ind)
#         needed_prob = nucleation_probabilities.get_probabilities(arr_len_in_flat, seeds[:, 2])
#         # print(seeds)
#         needed_prob[homogeneous_ind] = nucleation_probabilities.nucl_prob.values_pp[seeds[homogeneous_ind, 2]]
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
#     shm_p.close()
#     shm_a.close()
#     shm_product_init.close()
#     shm_product_x_nzs.close()


def ci_single(cur_case, seeds, oxidant, full_3d):

    shm_p = shared_memory.SharedMemory(name=cur_case.product_c3d_shm_mdata.name)
    product = np.ndarray(cur_case.product_c3d_shm_mdata.shape, dtype=cur_case.product_c3d_shm_mdata.dtype, buffer=shm_p.buf)
    shm_a = shared_memory.SharedMemory(name=cur_case.active_c3d_shm_mdata.name)
    active = np.ndarray(cur_case.active_c3d_shm_mdata.shape, dtype=cur_case.active_c3d_shm_mdata.dtype, buffer=shm_a.buf)
    shm_product_init = shared_memory.SharedMemory(name=cur_case.precip_3d_init_shm_mdata.name)
    product_init = np.ndarray(cur_case.precip_3d_init_shm_mdata.shape, dtype=cur_case.precip_3d_init_shm_mdata.dtype,
                              buffer=shm_product_init.buf)
    shm_product_x_nzs = shared_memory.SharedMemory(name=cur_case.prod_indexes_shm_mdata.name)
    product_x_nzs = np.ndarray(cur_case.prod_indexes_shm_mdata.shape, dtype=cur_case.prod_indexes_shm_mdata.dtype,
                               buffer=shm_product_x_nzs.buf)

    all_arounds = calc_sur_ind_formation(seeds, active.shape[2] - 1)
    neighbours = go_around_bool(active, all_arounds[:, :-2])
    arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.short)
    temp_ind = np.where(arr_len_out > 0)[0]

    if len(temp_ind) > 0:
        seeds = seeds[temp_ind]
        neighbours = neighbours[temp_ind]
        all_arounds = all_arounds[temp_ind]

        flat_arounds = np.concatenate((all_arounds[:, 0:5], all_arounds[:, -2:]), axis=1)
        arr_len_in_flat = cur_case.go_around_func_ref(product_init, flat_arounds)

        homogeneous_ind = np.where(arr_len_in_flat == 0)[0]
        needed_prob = cur_case.nucleation_probabilities.get_probabilities(arr_len_in_flat, seeds[:, 2])
        needed_prob[homogeneous_ind] = cur_case.nucleation_probabilities.nucl_prob.values_pp[seeds[homogeneous_ind, 2]]
        randomise = np.array(np.random.random_sample(arr_len_in_flat.size), dtype=np.float64)
        temp_ind = np.where(randomise < needed_prob)[0]

        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]

            out_to_del = [np.array(np.nonzero(item)[0]) for item in neighbours]
            to_del = [np.random.choice(item, 1, replace=False) for item in out_to_del]
            coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                             dtype=np.short)

            coord = np.reshape(coord, (len(coord) * 1, 3))
            coord = coord.transpose()
            seeds = seeds.transpose()

            active[coord[0], coord[1], coord[2]] -= 1
            oxidant[seeds[0], seeds[1], seeds[2]] -= 1

            # self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
            product[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

            # self.objs[self.case]["product"].fix_full_cells(coord)  # precip on place of active!
            # self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!

            cur_case.fix_full_cells(product, full_3d, seeds, cur_case.oxidation_number)

            # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
            # dissolution function
            product_x_nzs[seeds[2][0]] = True
    shm_p.close()
    shm_a.close()
    shm_product_init.close()
    shm_product_x_nzs.close()


def precip_step_two_products(cur_case, plane_indexes, fetch_indexes, callback):

    shm_o = shared_memory.SharedMemory(name=cur_case.oxidant_c3d_shm_mdata.name)
    oxidant = np.ndarray(cur_case.oxidant_c3d_shm_mdata.shape, dtype=cur_case.oxidant_c3d_shm_mdata.dtype,
                         buffer=shm_o.buf)

    shm_p_FULL = shared_memory.SharedMemory(name=cur_case.full_shm_mdata.name)
    full_3d = np.ndarray(cur_case.full_shm_mdata.shape, dtype=cur_case.full_shm_mdata.dtype, buffer=shm_p_FULL.buf)

    shm_to_check_with = shared_memory.SharedMemory(name=cur_case.to_check_with_shm_mdata.name)
    to_check_with = np.ndarray(cur_case.to_check_with_shm_mdata.shape, dtype=cur_case.to_check_with_shm_mdata.dtype,
                               buffer=shm_to_check_with.buf)

    for fetch_ind in fetch_indexes:
        plane_indexes = np.array(plane_indexes)
        nonzero_indices = np.nonzero(oxidant[fetch_ind[0][:, np.newaxis], fetch_ind[1][:, np.newaxis], plane_indexes])
        oxidant_cells = fetch_ind[:, nonzero_indices[0]]

        if len(oxidant_cells[0]) != 0:
            oxidant_cells = np.vstack((oxidant_cells, plane_indexes[np.array(nonzero_indices[1])]))
            oxidant_cells = np.array(oxidant_cells, dtype=np.short).transpose()

            exists = check_at_coord(full_3d, oxidant_cells)  # precip on place of oxidant!
            temp_ind = np.where(exists)[0]
            oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

            exists = check_at_coord(to_check_with, oxidant_cells)  # precip on place of oxidant!
            temp_ind = np.where(exists)[0]
            oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

            if len(oxidant_cells) > 0:
                # activate if microstructure ___________________________________________________________
                # in_gb = [self.microstructure[point[0], point[1], point[2]] for point in oxidant_cells]
                # temp_ind = np.where(in_gb)[0]
                # oxidant_cells = oxidant_cells[temp_ind]
                # ______________________________________________________________________________________
                callback(cur_case, oxidant_cells, oxidant, full_3d)

    shm_o.close()
    shm_p_FULL.close()
    shm_to_check_with.close()


def go_around_mult_oxid_n_also_partial_neigh_aip_MP(array_3d, around_coords):
    return np.sum(go_around_int(array_3d, around_coords), axis=1)


# def fix_full_cells(array_3d, full_array_3d, new_precip):
#     current_precip = np.array(array_3d[new_precip[0], new_precip[1], new_precip[2]], dtype=np.ubyte)
#     indexes = np.where(current_precip == 1)[0]
#     full_precip = new_precip[:, indexes]
#     full_array_3d[full_precip[0], full_precip[1], full_precip[2]] = True

