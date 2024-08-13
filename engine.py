from cellular_automata import *
from utils import data_base
import progressbar
import time
import keyboard


class SimulationConfigurator:
    """
    TODO: 1. Host elements in the CaseRef class instead of CellularAutomata!
          2. Buffer reserve for cells array to Config!!
          3. DEFAULT_PARAMS in templates, move out or create different script for cases!
    """
    def __init__(self):
        self.ca = CellularAutomata()
        self.utils = utils.Utils()

        self.db = data_base.Database()
        self.begin = None
        self.elapsed_time = None

        # setting objects for inward diffusion
        if Config.INWARD_DIFFUSION:
            self.init_inward()
        # setting objects for outward diffusion
        if Config.OUTWARD_DIFFUSION:
            self.init_outward()

        if Config.COMPUTE_PRECIPITATION:
            self.init_product()
            # self.ca.primary_product = elements.Product(Config.PRODUCTS.PRIMARY)
            # self.ca.cases.first.product = self.ca.primary_product
            #
            # if Config.ACTIVES.SECONDARY_EXISTENCE and Config.OXIDANTS.SECONDARY_EXISTENCE:
            #     print("NO IMPLEMENTATION YET")
            #
            # elif Config.ACTIVES.SECONDARY_EXISTENCE and not Config.OXIDANTS.SECONDARY_EXISTENCE:
            #
            #
            #     self.ca.secondary_product = elements.Product(Config.PRODUCTS.SECONDARY)
            #     self.ca.cases.second.product = self.ca.secondary_product
            #     self.ca.cases.first.to_check_with = self.ca.secondary_product
            #     self.ca.cases.second.to_check_with = self.ca.primary_product
            #
            #     self.ca.cases.first.prod_indexes = np.full(Config.N_CELLS_PER_AXIS, False, dtype=bool)
            #     self.ca.cases.second.prod_indexes = np.full(Config.N_CELLS_PER_AXIS, False, dtype=bool)
            #
            #     if self.ca.cases.first.product.oxidation_number == 1:
            #         self.ca.cases.first.go_around_func_ref = self.ca.go_around_single_oxid_n
            #         self.ca.cases.first.fix_init_precip_func_ref = self.ca.fix_init_precip_bool
            #         my_type = bool
            #     else:
            #         self.ca.cases.first.go_around_func_ref = self.ca.go_around_mult_oxid_n
            #         self.ca.cases.first.fix_init_precip_func_ref = self.ca.fix_init_precip_int
            #         my_type = np.ubyte
            #     self.ca.cases.first.precip_3d_init = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS,
            #                                                   Config.N_CELLS_PER_AXIS + 1), 0, dtype=my_type)
            #
            #     if self.ca.cases.second.product.oxidation_number == 1:
            #         self.ca.cases.second.go_around_func_ref = self.ca.go_around_single_oxid_n
            #         self.ca.cases.second.fix_init_precip_func_ref = self.ca.fix_init_precip_bool
            #         my_type = bool
            #     else:
            #         self.ca.cases.second.go_around_func_ref = self.ca.go_around_mult_oxid_n
            #         self.ca.cases.second.fix_init_precip_func_ref = self.ca.fix_init_precip_int
            #         my_type = np.ubyte
            #     self.ca.cases.second.precip_3d_init = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS,
            #                                                   Config.N_CELLS_PER_AXIS + 1), 0, dtype=my_type)
            # else:
            #     self.ca.primary_oxidant.scale = self.ca.primary_product
            #     self.ca.primary_active.scale = self.ca.primary_product
            #
            #     if self.ca.cases.first.product.oxidation_number == 1:
            #         self.ca.cases.first.go_around_func_ref = self.ca.go_around_single_oxid_n
            #         self.ca.cases.first.fix_init_precip_func_ref = self.ca.fix_init_precip_bool
            #         self.ca.cases.first.precip_3d_init = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS,
            #                                                       Config.N_CELLS_PER_AXIS + 1), False, dtype=bool)
            #     else:
            #         # self.cases.first.go_around_func_ref = self.go_around_mult_oxid_n
            #         self.ca.cases.first.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
            #         # self.cases.first.go_around_func_ref = self.go_around_mult_oxid_n_also_partial_neigh  # CHANGE!!!!
            #         # self.cases.first.fix_init_precip_func_ref = self.fix_init_precip_int
            #         self.ca.cases.first.precip_3d_init = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS,
            #                                                       Config.N_CELLS_PER_AXIS + 1), 0, dtype=np.ubyte)

    def configurate_functions(self):
        self.ca.primary_oxidant.diffuse = self.ca.primary_oxidant.diffuse_bulk
        self.ca.primary_active.diffuse = elements.diffuse_bulk_mp

        self.ca.get_cur_ioz_bound = self.ca.ioz_depth_furthest_inward

        self.ca.precip_func = self.ca.precipitation_first_case
        self.ca.get_combi_ind = self.ca.get_combi_ind_standard

        self.ca.cases.first_mp.precip_step = precip_step_standard
        self.ca.cases.first_mp.check_intersection = ci_single

        self.ca.decomposition = self.ca.dissolution_atomic_stop_if_stable
        self.ca.decomposition_intrinsic = self.ca.simple_decompose_mp
        self.ca.cases.first_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL

        self.ca.cur_case = self.ca.cases.first
        self.ca.cur_case_mp = self.ca.cases.first_mp
        # self.ca.cases.first.fix_init_precip_func_ref = self.ca.fix_init_precip_dummy
        # self.ca.cases.first.go_around_func_ref = self.ca.go_around_mult_oxid_n_also_partial_neigh_aip_MP

        self.ca.cases.first_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                  Config.PRODUCTS.PRIMARY)
        self.ca.cases.first_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                    Config.PRODUCTS.PRIMARY)

    def functions_sec_case(self):
        self.ca.primary_oxidant.diffuse = self.ca.primary_oxidant.diffuse_bulk
        self.ca.primary_active.diffuse = elements.diffuse_bulk_mp
        self.ca.secondary_active.diffuse = elements.diffuse_bulk_mp

        self.ca.get_cur_ioz_bound = self.ca.ioz_depth_from_kinetics

        self.ca.precip_func = self.ca.precipitation_second_case
        self.ca.get_combi_ind = self.ca.get_combi_ind_atomic_two_products

        self.ca.cases.first_mp.precip_step = precip_step_multi_products
        self.ca.cases.first_mp.check_intersection = ci_single

        self.ca.cases.second_mp.precip_step = precip_step_multi_products
        self.ca.cases.second_mp.check_intersection = ci_single

        self.ca.decomposition = self.ca.dissolution_atomic_stop_if_stable_two_products
        self.ca.decomposition_intrinsic = self.ca.simple_decompose_mp

        self.ca.cases.first_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL
        self.ca.cases.second_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL

        self.ca.cases.first_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                        Config.PRODUCTS.PRIMARY)
        self.ca.cases.first_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                          Config.PRODUCTS.PRIMARY)

        self.ca.cases.second_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.SECONDARY,
                                                                                  Config.PRODUCTS.SECONDARY)
        self.ca.cases.second_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.SECONDARY,
                                                                                    Config.PRODUCTS.SECONDARY)

    def configurate_functions_td(self):
        self.ca.primary_oxidant.diffuse = self.ca.primary_oxidant.diffuse_bulk
        self.ca.primary_active.diffuse = elements.diffuse_bulk_mp
        self.ca.secondary_active.diffuse = elements.diffuse_bulk_mp

        self.ca.precip_func = self.ca.precipitation_with_td
        self.ca.get_combi_ind = None

        self.ca.get_cur_ioz_bound = self.ca.ioz_depth_from_kinetics

        self.ca.cases.first_mp.precip_step = precip_step_multi_products
        self.ca.cases.first_mp.check_intersection = ci_single

        self.ca.cases.second_mp.precip_step = precip_step_multi_products
        self.ca.cases.second_mp.check_intersection = ci_single

        self.ca.decomposition = None
        self.ca.decomposition_intrinsic = self.ca.simple_decompose_mp

        self.ca.cases.first_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL
        self.ca.cases.second_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL

        self.ca.cases.first_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                        Config.PRODUCTS.PRIMARY)
        self.ca.cases.first_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                          Config.PRODUCTS.PRIMARY)
        self.ca.cases.second_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.SECONDARY,
                                                                                         Config.PRODUCTS.SECONDARY)
        self.ca.cases.second_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.SECONDARY,
                                                                                           Config.PRODUCTS.SECONDARY)

    def configurate_functions_td_all(self):
        self.ca.primary_oxidant.diffuse = self.ca.primary_oxidant.diffuse_with_scale
        self.ca.primary_active.diffuse = elements.diffuse_bulk_mp
        self.ca.secondary_active.diffuse = elements.diffuse_bulk_mp

        self.ca.precip_func = self.ca.precipitation_with_td
        self.ca.get_combi_ind = None

        self.ca.get_cur_ioz_bound = self.ca.ioz_depth_furthest_inward

        self.ca.cases.first_mp.precip_step = precip_step_multi_products
        self.ca.cases.first_mp.check_intersection = ci_multi

        self.ca.cases.second_mp.precip_step = precip_step_multi_products
        self.ca.cases.second_mp.check_intersection = ci_multi

        self.ca.cases.third_mp.precip_step = precip_step_multi_products
        self.ca.cases.third_mp.check_intersection = ci_multi

        self.ca.cases.fourth_mp.precip_step = precip_step_multi_products
        self.ca.cases.fourth_mp.check_intersection = ci_multi

        self.ca.cases.fifth_mp.precip_step = precip_step_multi_products
        self.ca.cases.fifth_mp.check_intersection = ci_multi_no_active

        self.ca.decomposition = None
        self.ca.decomposition_intrinsic = self.ca.simple_decompose_mp

        self.ca.cases.first_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL
        self.ca.cases.second_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL
        self.ca.cases.third_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL
        self.ca.cases.fourth_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL
        self.ca.cases.fifth_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL

        self.ca.cases.first_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                        Config.PRODUCTS.PRIMARY)
        self.ca.cases.first_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                          Config.PRODUCTS.PRIMARY)

        self.ca.cases.second_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.SECONDARY,
                                                                                         Config.PRODUCTS.SECONDARY)
        self.ca.cases.second_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.SECONDARY,
                                                                                           Config.PRODUCTS.SECONDARY)

        self.ca.cases.third_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.TERNARY,
                                                                                         Config.PRODUCTS.TERNARY)
        self.ca.cases.third_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.TERNARY,
                                                                                          Config.PRODUCTS.TERNARY)

        self.ca.cases.fourth_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.QUATERNARY,
                                                                                         Config.PRODUCTS.QUATERNARY)
        self.ca.cases.fourth_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.QUATERNARY,
                                                                                           Config.PRODUCTS.QUATERNARY)

        self.ca.cases.fifth_mp.nucleation_probabilities = utils.NucleationProbabilities(
            Config.PROBABILITIES.QUINT,
            Config.PRODUCTS.QUINT)
        self.ca.cases.fifth_mp.dissolution_probabilities = utils.DissolutionProbabilities(
            Config.PROBABILITIES.QUINT,
            Config.PRODUCTS.QUINT)

    def run_simulation(self):
        self.begin = time.time()
        for self.ca.iteration in progressbar.progressbar(range(Config.N_ITERATIONS)):
            if keyboard.is_pressed('ctrl+g+m'):
                break
            self.ca.precip_func()
            # self.ca.decomposition()
            self.ca.diffusion_inward()
            self.ca.diffusion_outward()
            # self.calc_precipitation_front_only_cells()
            # self.diffusion_outward_with_mult_srtide()

        end = time.time()
        self.elapsed_time = (end - self.begin)
        self.db.insert_time(self.elapsed_time)
        self.db.conn.commit()

    def init_inward(self):
        self.ca.primary_oxidant = elements.OxidantElem(Config.OXIDANTS.PRIMARY, self.utils)
        self.ca.cases.first.oxidant = self.ca.primary_oxidant
        self.ca.cases.second.oxidant = self.ca.primary_oxidant

        self.ca.cases.first_mp.oxidant_c3d_shm_mdata = self.ca.primary_oxidant.c3d_shm_mdata
        self.ca.cases.second_mp.oxidant_c3d_shm_mdata = self.ca.primary_oxidant.c3d_shm_mdata

        self.ca.cases.third.oxidant = self.ca.primary_oxidant
        self.ca.cases.fourth.oxidant = self.ca.primary_oxidant
        self.ca.cases.fifth.oxidant = self.ca.primary_oxidant

        self.ca.cases.third_mp.oxidant_c3d_shm_mdata = self.ca.primary_oxidant.c3d_shm_mdata
        self.ca.cases.fourth_mp.oxidant_c3d_shm_mdata = self.ca.primary_oxidant.c3d_shm_mdata
        self.ca.cases.fifth_mp.oxidant_c3d_shm_mdata = self.ca.primary_oxidant.c3d_shm_mdata

        # ---------------------------------------------------
        if Config.OXIDANTS.SECONDARY_EXISTENCE:
            self.ca.secondary_oxidant = elements.OxidantElem(Config.OXIDANTS.SECONDARY, self.utils)
            self.ca.cases.third.oxidant = self.ca.secondary_oxidant
            self.ca.cases.fourth.oxidant = self.ca.secondary_oxidant

            self.ca.cases.third_mp.oxidant_c3d_shm_mdata = self.ca.secondary_oxidant.c3d_shm_mdata
            self.ca.cases.fourth_mp.oxidant_c3d_shm_mdata = self.ca.secondary_oxidant.c3d_shm_mdata

    def init_outward(self):
        self.ca.primary_active = elements.ActiveElem(Config.ACTIVES.PRIMARY)
        self.ca.cases.first.active = self.ca.primary_active
        self.ca.cases.third.active = self.ca.primary_active
        # ---------------------------------------------------
        # c3d
        self.ca.cases.first_mp.active_c3d_shm_mdata = self.ca.primary_active.c3d_shm_mdata
        self.ca.cases.third_mp.active_c3d_shm_mdata = self.ca.primary_active.c3d_shm_mdata

        self.ca.cases.fifth_mp.active_c3d_shm_mdata = self.ca.primary_active.c3d_shm_mdata # JUST FOR SHAPE!!!

        # cells
        self.ca.cases.first_mp.active_cells_shm_mdata = self.ca.primary_active.cells_shm_mdata
        self.ca.cases.third_mp.active_cells_shm_mdata = self.ca.primary_active.cells_shm_mdata
        # dirs
        self.ca.cases.first_mp.active_dirs_shm_mdata = self.ca.primary_active.dirs_shm_mdata
        self.ca.cases.third_mp.active_dirs_shm_mdata = self.ca.primary_active.dirs_shm_mdata

        # ---------------------------------------------------
        if Config.ACTIVES.SECONDARY_EXISTENCE:
            self.ca.secondary_active = elements.ActiveElem(Config.ACTIVES.SECONDARY)
            self.ca.cases.second.active = self.ca.secondary_active
            self.ca.cases.fourth.active = self.ca.secondary_active
            # ---------------------------------------------------
            # c3d
            self.ca.cases.second_mp.active_c3d_shm_mdata = self.ca.secondary_active.c3d_shm_mdata
            self.ca.cases.fourth_mp.active_c3d_shm_mdata = self.ca.secondary_active.c3d_shm_mdata
            # cells
            self.ca.cases.second_mp.active_cells_shm_mdata = self.ca.secondary_active.cells_shm_mdata
            self.ca.cases.fourth_mp.active_cells_shm_mdata = self.ca.secondary_active.cells_shm_mdata
            # dirs
            self.ca.cases.second_mp.active_dirs_shm_mdata = self.ca.secondary_active.dirs_shm_mdata
            self.ca.cases.fourth_mp.active_dirs_shm_mdata = self.ca.secondary_active.dirs_shm_mdata

    def init_product(self):

        if Config.ACTIVES.SECONDARY_EXISTENCE and Config.OXIDANTS.SECONDARY_EXISTENCE:
            # accumulated products
            tmp = np.zeros((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), dtype=np.ubyte)
            self.ca.cases.accumulated_products_shm = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
            self.ca.cases.accumulated_products = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                            buffer=self.ca.cases.accumulated_products_shm.buf)
            np.copyto(self.ca.cases.accumulated_products, tmp)
            self.ca.cases.accumulated_products_shm_mdata = SharedMetaData(self.ca.cases.accumulated_products_shm.name,
                                                                          tmp.shape, tmp.dtype)
            self.init_first_case()
            self.init_second_case()
            self.init_third_case()
            self.init_fourth_case()
            self.init_fifth_case()

        elif Config.ACTIVES.SECONDARY_EXISTENCE and not Config.OXIDANTS.SECONDARY_EXISTENCE:
            # accumulated products
            tmp = np.zeros((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1),
                           dtype=np.ubyte)
            self.ca.cases.accumulated_products_shm = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
            self.ca.cases.accumulated_products = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                            buffer=self.ca.cases.accumulated_products_shm.buf)
            np.copyto(self.ca.cases.accumulated_products, tmp)
            self.ca.cases.accumulated_products_shm_mdata = SharedMetaData(self.ca.cases.accumulated_products_shm.name,
                                                                          tmp.shape, tmp.dtype)
            self.init_first_case()
            self.init_second_case()
            self.init_third_case()
            self.init_fourth_case()
            self.init_fifth_case()
        else:
            self.init_first_case()

    def init_first_case(self):
        self.ca.cases.first_mp.cells_per_axis = self.ca.cells_per_axis
        self.ca.primary_product = elements.Product(Config.PRODUCTS.PRIMARY)

        self.ca.cases.first.product = self.ca.primary_product
        self.ca.cases.first_mp.oxidation_number = self.ca.primary_product.oxidation_number

        self.ca.cases.first_mp.threshold_inward = Config.PRODUCTS.PRIMARY.THRESHOLD_INWARD
        self.ca.cases.first_mp.threshold_outward = Config.PRODUCTS.PRIMARY.THRESHOLD_OUTWARD

        # to check with
        self.ca.cases.first_mp.to_check_with_shm_mdata = self.ca.cases.accumulated_products_shm_mdata
        # scale
        self.ca.primary_oxidant.scale = self.ca.cases.accumulated_products
        self.ca.primary_active.scale = self.ca.cases.accumulated_products
        # c3d
        self.ca.cases.first_mp.product_c3d_shm_mdata = self.ca.primary_product.c3d_shm_mdata
        # full_c3d
        self.ca.cases.first_mp.full_shm_mdata = self.ca.primary_product.full_shm_mdata
        # product indexes
        tmp = np.full(Config.N_CELLS_PER_AXIS, False, dtype=bool)
        self.ca.cases.first.shm_pool["product_indexes"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.first.prod_indexes = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.ca.cases.first.shm_pool["product_indexes"].buf)
        np.copyto(self.ca.cases.first.prod_indexes, tmp)
        self.ca.cases.first_mp.prod_indexes_shm_mdata = SharedMetaData(self.ca.cases.first.shm_pool["product_indexes"].name,
                                                                       tmp.shape, tmp.dtype)
        # ind not stable
        tmp = np.full(Config.N_CELLS_PER_AXIS, True, dtype=bool)
        self.ca.cases.first.shm_pool["product_ind_not_stab"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.first.product_ind_not_stab = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                              buffer=self.ca.cases.first.shm_pool["product_ind_not_stab"].buf)
        np.copyto(self.ca.cases.first.product_ind_not_stab, tmp)
        self.ca.cases.first_mp.prod_indexes_not_stab_shm_mdata = SharedMetaData(self.ca.cases.first.shm_pool["product_ind_not_stab"].name,
                                                                                tmp.shape, tmp.dtype)
        # functions
        if self.ca.cases.first.product.oxidation_number == 1:
            self.ca.cases.first_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
            self.ca.cases.first.fix_init_precip_func_ref = self.ca.fix_init_precip_bool
            my_type = bool
        else:
            self.ca.cases.first_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
            self.ca.cases.first.fix_init_precip_func_ref = self.ca.fix_init_precip_int
            my_type = np.ubyte
        # c3d_init
        tmp = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), False,
                      dtype=my_type)
        self.ca.cases.first.shm_pool["precip_3d_init"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.first.precip_3d_init = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.ca.cases.first.shm_pool["precip_3d_init"].buf)
        np.copyto(self.ca.cases.first.precip_3d_init, tmp)
        self.ca.cases.first_mp.precip_3d_init_shm_mdata = SharedMetaData(self.ca.cases.first.shm_pool["precip_3d_init"].name,
                                                                         tmp.shape, tmp.dtype)
        # fix full cells
        self.ca.cases.first_mp.fix_full_cells = elements.fix_full_cells

    def init_second_case(self):
        self.ca.cases.second_mp.cells_per_axis = self.ca.cells_per_axis
        self.ca.secondary_product = elements.Product(Config.PRODUCTS.SECONDARY)
        self.ca.cases.second.product = self.ca.secondary_product
        self.ca.cases.second_mp.oxidation_number = self.ca.secondary_product.oxidation_number

        self.ca.cases.second_mp.threshold_inward = Config.PRODUCTS.SECONDARY.THRESHOLD_INWARD
        self.ca.cases.second_mp.threshold_outward = Config.PRODUCTS.SECONDARY.THRESHOLD_OUTWARD

        # to check with
        self.ca.cases.second_mp.to_check_with_shm_mdata = self.ca.cases.accumulated_products_shm_mdata

        # c3d
        self.ca.cases.second_mp.product_c3d_shm_mdata = self.ca.secondary_product.c3d_shm_mdata
        # full_c3d
        self.ca.cases.second_mp.full_shm_mdata = self.ca.secondary_product.full_shm_mdata
        # product indexes
        tmp = np.full(Config.N_CELLS_PER_AXIS, False, dtype=bool)
        self.ca.cases.second.shm_pool["product_indexes"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.second.prod_indexes = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                      buffer=self.ca.cases.second.shm_pool["product_indexes"].buf)
        np.copyto(self.ca.cases.second.prod_indexes, tmp)
        self.ca.cases.second_mp.prod_indexes_shm_mdata = SharedMetaData(self.ca.cases.second.shm_pool["product_indexes"].name,
                                                                        tmp.shape, tmp.dtype)
        # ind not stable
        tmp = np.full(Config.N_CELLS_PER_AXIS, True, dtype=bool)
        self.ca.cases.second.shm_pool["product_ind_not_stab"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.second.product_ind_not_stab = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                               buffer=self.ca.cases.second.shm_pool["product_ind_not_stab"].buf)
        np.copyto(self.ca.cases.second.product_ind_not_stab, tmp)
        self.ca.cases.second_mp.prod_indexes_not_stab_shm_mdata = SharedMetaData(self.ca.cases.second.shm_pool["product_ind_not_stab"].name,
                                                                                 tmp.shape, tmp.dtype)
        # c3d_init
        if self.ca.cases.second.product.oxidation_number == 1:
            self.ca.cases.second_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
            self.ca.cases.second.fix_init_precip_func_ref = self.ca.fix_init_precip_bool
            my_type = bool
        else:
            self.ca.cases.second_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
            self.ca.cases.second.fix_init_precip_func_ref = self.ca.fix_init_precip_int
            my_type = np.ubyte

        tmp = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), False,
                      dtype=my_type)
        self.ca.cases.second.shm_pool["precip_3d_init"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.second.precip_3d_init = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                        buffer=self.ca.cases.second.shm_pool["precip_3d_init"].buf)
        np.copyto(self.ca.cases.second.precip_3d_init, tmp)
        self.ca.cases.second_mp.precip_3d_init_shm_mdata = SharedMetaData(self.ca.cases.second.shm_pool["precip_3d_init"].name,
                                                                          tmp.shape, tmp.dtype)
        # fix full cells
        self.ca.cases.second_mp.fix_full_cells = elements.fix_full_cells

    def init_third_case(self):
        self.ca.cases.third_mp.cells_per_axis = self.ca.cells_per_axis
        self.ca.ternary_product = elements.Product(Config.PRODUCTS.TERNARY)
        self.ca.cases.third.product = self.ca.ternary_product
        self.ca.cases.third_mp.oxidation_number = self.ca.ternary_product.oxidation_number

        self.ca.cases.third_mp.threshold_inward = Config.PRODUCTS.TERNARY.THRESHOLD_INWARD
        self.ca.cases.third_mp.threshold_outward = Config.PRODUCTS.TERNARY.THRESHOLD_OUTWARD

        # to check with
        self.ca.cases.third_mp.to_check_with_shm_mdata = self.ca.cases.accumulated_products_shm_mdata

        # c3d
        self.ca.cases.third_mp.product_c3d_shm_mdata = self.ca.ternary_product.c3d_shm_mdata
        # full_c3d
        self.ca.cases.third_mp.full_shm_mdata = self.ca.ternary_product.full_shm_mdata
        # product indexes
        tmp = np.full(Config.N_CELLS_PER_AXIS, False, dtype=bool)
        self.ca.cases.third.shm_pool["product_indexes"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.third.prod_indexes = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                       buffer=self.ca.cases.third.shm_pool["product_indexes"].buf)
        np.copyto(self.ca.cases.third.prod_indexes, tmp)
        self.ca.cases.third_mp.prod_indexes_shm_mdata = SharedMetaData(
            self.ca.cases.third.shm_pool["product_indexes"].name,
            tmp.shape, tmp.dtype)
        # ind not stable
        tmp = np.full(Config.N_CELLS_PER_AXIS, True, dtype=bool)
        self.ca.cases.third.shm_pool["product_ind_not_stab"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.third.product_ind_not_stab = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                               buffer=self.ca.cases.third.shm_pool[
                                                                   "product_ind_not_stab"].buf)
        np.copyto(self.ca.cases.third.product_ind_not_stab, tmp)
        self.ca.cases.third_mp.prod_indexes_not_stab_shm_mdata = SharedMetaData(
            self.ca.cases.third.shm_pool["product_ind_not_stab"].name,
            tmp.shape, tmp.dtype)
        # c3d_init
        if self.ca.cases.third.product.oxidation_number == 1:
            self.ca.cases.third_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
            self.ca.cases.third.fix_init_precip_func_ref = self.ca.fix_init_precip_bool
            my_type = bool
        else:
            self.ca.cases.third_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
            self.ca.cases.third.fix_init_precip_func_ref = self.ca.fix_init_precip_int
            my_type = np.ubyte

        tmp = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), False,
                      dtype=my_type)
        self.ca.cases.third.shm_pool["precip_3d_init"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.third.precip_3d_init = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                         buffer=self.ca.cases.third.shm_pool["precip_3d_init"].buf)
        np.copyto(self.ca.cases.third.precip_3d_init, tmp)
        self.ca.cases.third_mp.precip_3d_init_shm_mdata = SharedMetaData(
            self.ca.cases.third.shm_pool["precip_3d_init"].name,
            tmp.shape, tmp.dtype)
        # fix full cells
        self.ca.cases.third_mp.fix_full_cells = elements.fix_full_cells

    def init_fourth_case(self):
        self.ca.cases.fourth_mp.cells_per_axis = self.ca.cells_per_axis
        self.ca.quaternary_product = elements.Product(Config.PRODUCTS.QUATERNARY)
        self.ca.cases.fourth.product = self.ca.quaternary_product
        self.ca.cases.fourth_mp.oxidation_number = self.ca.quaternary_product.oxidation_number

        self.ca.cases.fourth_mp.threshold_inward = Config.PRODUCTS.QUATERNARY.THRESHOLD_INWARD
        self.ca.cases.fourth_mp.threshold_outward = Config.PRODUCTS.QUATERNARY.THRESHOLD_OUTWARD

        # to check with
        self.ca.cases.fourth_mp.to_check_with_shm_mdata = self.ca.cases.accumulated_products_shm_mdata

        # c3d
        self.ca.cases.fourth_mp.product_c3d_shm_mdata = self.ca.quaternary_product.c3d_shm_mdata
        # full_c3d
        self.ca.cases.fourth_mp.full_shm_mdata = self.ca.quaternary_product.full_shm_mdata
        # product indexes
        tmp = np.full(Config.N_CELLS_PER_AXIS, False, dtype=bool)
        self.ca.cases.fourth.shm_pool["product_indexes"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.fourth.prod_indexes = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                       buffer=self.ca.cases.fourth.shm_pool["product_indexes"].buf)
        np.copyto(self.ca.cases.fourth.prod_indexes, tmp)
        self.ca.cases.fourth_mp.prod_indexes_shm_mdata = SharedMetaData(
            self.ca.cases.fourth.shm_pool["product_indexes"].name,
            tmp.shape, tmp.dtype)
        # ind not stable
        tmp = np.full(Config.N_CELLS_PER_AXIS, True, dtype=bool)
        self.ca.cases.fourth.shm_pool["product_ind_not_stab"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.fourth.product_ind_not_stab = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                               buffer=self.ca.cases.fourth.shm_pool[
                                                                   "product_ind_not_stab"].buf)
        np.copyto(self.ca.cases.fourth.product_ind_not_stab, tmp)
        self.ca.cases.fourth_mp.prod_indexes_not_stab_shm_mdata = SharedMetaData(
            self.ca.cases.fourth.shm_pool["product_ind_not_stab"].name,
            tmp.shape, tmp.dtype)
        # c3d_init
        if self.ca.cases.fourth.product.oxidation_number == 1:
            self.ca.cases.fourth_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
            self.ca.cases.fourth.fix_init_precip_func_ref = self.ca.fix_init_precip_bool
            my_type = bool
        else:
            self.ca.cases.fourth_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
            self.ca.cases.fourth.fix_init_precip_func_ref = self.ca.fix_init_precip_int
            my_type = np.ubyte

        tmp = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), False,
                      dtype=my_type)
        self.ca.cases.fourth.shm_pool["precip_3d_init"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.fourth.precip_3d_init = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                         buffer=self.ca.cases.fourth.shm_pool["precip_3d_init"].buf)
        np.copyto(self.ca.cases.fourth.precip_3d_init, tmp)
        self.ca.cases.fourth_mp.precip_3d_init_shm_mdata = SharedMetaData(
            self.ca.cases.fourth.shm_pool["precip_3d_init"].name,
            tmp.shape, tmp.dtype)
        # fix full cells
        self.ca.cases.fourth_mp.fix_full_cells = elements.fix_full_cells

    def init_fifth_case(self):
        self.ca.cases.fifth_mp.cells_per_axis = self.ca.cells_per_axis
        self.ca.quint_product = elements.Product(Config.PRODUCTS.QUINT)
        self.ca.cases.fifth.product = self.ca.quint_product
        self.ca.cases.fifth_mp.oxidation_number = self.ca.quint_product.oxidation_number

        self.ca.cases.fifth_mp.threshold_inward = Config.PRODUCTS.QUINT.THRESHOLD_INWARD
        self.ca.cases.fifth_mp.threshold_outward = Config.PRODUCTS.QUINT.THRESHOLD_OUTWARD

        # to check with
        self.ca.cases.fifth_mp.to_check_with_shm_mdata = self.ca.cases.accumulated_products_shm_mdata

        # c3d
        self.ca.cases.fifth_mp.product_c3d_shm_mdata = self.ca.quint_product.c3d_shm_mdata
        # full_c3d
        self.ca.cases.fifth_mp.full_shm_mdata = self.ca.quint_product.full_shm_mdata
        # product indexes
        tmp = np.full(Config.N_CELLS_PER_AXIS, False, dtype=bool)
        self.ca.cases.fifth.shm_pool["product_indexes"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.fifth.prod_indexes = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                       buffer=self.ca.cases.fifth.shm_pool["product_indexes"].buf)
        np.copyto(self.ca.cases.fifth.prod_indexes, tmp)
        self.ca.cases.fifth_mp.prod_indexes_shm_mdata = SharedMetaData(
            self.ca.cases.fifth.shm_pool["product_indexes"].name,
            tmp.shape, tmp.dtype)
        # ind not stable
        tmp = np.full(Config.N_CELLS_PER_AXIS, True, dtype=bool)
        self.ca.cases.fifth.shm_pool["product_ind_not_stab"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.fifth.product_ind_not_stab = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                               buffer=self.ca.cases.fifth.shm_pool[
                                                                   "product_ind_not_stab"].buf)
        np.copyto(self.ca.cases.fifth.product_ind_not_stab, tmp)
        self.ca.cases.fifth_mp.prod_indexes_not_stab_shm_mdata = SharedMetaData(
            self.ca.cases.fifth.shm_pool["product_ind_not_stab"].name,
            tmp.shape, tmp.dtype)
        # c3d_init
        if self.ca.cases.fifth.product.oxidation_number == 1:
            self.ca.cases.fifth_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
            self.ca.cases.fifth.fix_init_precip_func_ref = self.ca.fix_init_precip_bool
            my_type = bool
        else:
            self.ca.cases.fifth_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
            self.ca.cases.fifth.fix_init_precip_func_ref = self.ca.fix_init_precip_int
            my_type = np.ubyte

        tmp = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), False,
                      dtype=my_type)
        self.ca.cases.fifth.shm_pool["precip_3d_init"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.ca.cases.fifth.precip_3d_init = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                         buffer=self.ca.cases.fifth.shm_pool["precip_3d_init"].buf)
        np.copyto(self.ca.cases.fifth.precip_3d_init, tmp)
        self.ca.cases.fifth_mp.precip_3d_init_shm_mdata = SharedMetaData(
            self.ca.cases.fifth.shm_pool["precip_3d_init"].name,
            tmp.shape, tmp.dtype)
        # fix full cells
        self.ca.cases.fifth_mp.fix_full_cells = elements.fix_full_cells

    def save_results(self):
        if Config.STRIDE > Config.N_ITERATIONS:
            self.ca.primary_active.transform_to_descards()
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.ca.secondary_active.transform_to_descards()
        if Config.INWARD_DIFFUSION:
            self.db.insert_particle_data("primary_oxidant", self.ca.iteration, self.ca.primary_oxidant.cells)
            if Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_oxidant", self.ca.iteration, self.ca.secondary_oxidant.cells)
        if Config.OUTWARD_DIFFUSION:
            self.db.insert_particle_data("primary_active", self.ca.iteration, self.ca.primary_active.get_cells_coords())
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_active", self.ca.iteration, self.ca.secondary_active.get_cells_coords())
        if Config.COMPUTE_PRECIPITATION:
            self.db.insert_particle_data("primary_product", self.ca.iteration, self.ca.primary_product.transform_c3d())
            if Config.ACTIVES.SECONDARY_EXISTENCE and Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_product", self.ca.iteration,
                                                   self.ca.secondary_product.transform_c3d())
                self.db.insert_particle_data("ternary_product", self.ca.iteration,
                                                   self.ca.ternary_product.transform_c3d())
                self.db.insert_particle_data("quaternary_product", self.ca.iteration,
                                                   self.ca.quaternary_product.transform_c3d())
            elif Config.ACTIVES.SECONDARY_EXISTENCE and not Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_product", self.ca.iteration,
                                                   self.ca.secondary_product.transform_c3d())
        if Config.STRIDE > Config.N_ITERATIONS:
            self.ca.primary_active.transform_to_3d(self.ca.curr_max_furthest)
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.ca.secondary_active.transform_to_3d(self.ca.curr_max_furthest)

    def save_results_custom(self):
        if Config.STRIDE > Config.N_ITERATIONS:
            self.ca.primary_active.transform_to_descards()
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.ca.secondary_active.transform_to_descards()

        self.db.insert_particle_data("primary_oxidant", self.ca.iteration, self.ca.primary_oxidant.cells)

        self.db.insert_particle_data("primary_active", self.ca.iteration, self.ca.primary_active.get_cells_coords())
        self.db.insert_particle_data("secondary_active", self.ca.iteration, self.ca.secondary_active.get_cells_coords())

        self.db.insert_particle_data("primary_product", self.ca.iteration, self.ca.primary_product.transform_c3d())
        self.db.insert_particle_data("secondary_product", self.ca.iteration, self.ca.secondary_product.transform_c3d())
        self.db.insert_particle_data("ternary_product", self.ca.iteration, self.ca.ternary_product.transform_c3d())
        self.db.insert_particle_data("quaternary_product", self.ca.iteration, self.ca.quaternary_product.transform_c3d())
        self.db.insert_particle_data("quint_product", self.ca.iteration, self.ca.quint_product.transform_c3d())

        if Config.STRIDE > Config.N_ITERATIONS:
            self.ca.primary_active.transform_to_3d(self.ca.curr_max_furthest)
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.ca.secondary_active.transform_to_3d(self.ca.curr_max_furthest)

    def calc_precipitation_front_only_cells(self):
        """
        Calculating a position of a precipitation front, considering only cells concentrations without any scaling!
        As a boundary a product fraction of 0,1% is used.
        """
        product = np.array([np.sum(self.ca.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.ca.cells_per_axis)], dtype=np.uint32)
        product = product / (self.ca.cells_per_axis ** 2)
        threshold = Config.ACTIVES.PRIMARY.CELLS_CONCENTRATION
        for rev_index, precip_conc in enumerate(np.flip(product)):
            if precip_conc > threshold / 100:
                position = (len(product) - 1 - rev_index) * Config.SIZE * 10 ** 6 \
                           / self.ca.cells_per_axis
                sqr_time = ((self.ca.iteration + 1) * Config.SIM_TIME / (self.ca.n_iter * 3600)) ** (1 / 2)
                self.db.insert_precipitation_front(sqr_time, position, "p")
                break

    def terminate_workers(self):
        self.ca.pool.close()
        self.ca.pool.join()
        print()
        print("TERMINATED PROPERLY!")

    def unlink(self):
        # self.ca.precip_3d_init_shm.close()
        # self.ca.precip_3d_init_shm.unlink()

        # self.ca.product_x_nzs_shm.close()
        # self.ca.product_x_nzs_shm.unlink()

        self.ca.primary_active.c3d_shared.close()
        self.ca.primary_active.c3d_shared.unlink()

        self.ca.primary_oxidant.c3d_shared.close()
        self.ca.primary_oxidant.c3d_shared.unlink()

        self.ca.primary_product.c3d_shared.close()
        self.ca.primary_product.c3d_shared.unlink()

        self.ca.primary_product.full_c3d_shared.close()
        self.ca.primary_product.full_c3d_shared.unlink()

        self.ca.primary_active.cells_shm.close()
        self.ca.primary_active.cells_shm.unlink()

        self.ca.primary_active.dirs_shm.close()
        self.ca.primary_active.dirs_shm.unlink()

        self.ca.cases.close_shms()

        print("UNLINKED PROPERLY!")

    def save_results_only_prod(self):
        self.db.insert_particle_data("primary_product", self.ca.iteration,
                                           self.ca.primary_product.transform_c3d())

    def save_results_prod_and_inw(self):
        self.db.insert_particle_data("primary_product", self.ca.iteration,
                                           self.ca.primary_product.transform_c3d())
        self.db.insert_particle_data("primary_oxidant", self.ca.iteration, self.ca.primary_oxidant.cells)

    def save_results_only_inw(self):
        self.db.insert_particle_data("primary_oxidant", self.ca.iteration, self.ca.primary_oxidant.cells)

    def insert_last_it(self):
        self.db.insert_last_iteration(self.ca.iteration)
