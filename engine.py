from cellular_automata import *
from utils import data_base
import progressbar
import time
import keyboard
from microstructure import voronoi
import elements


class SimulationConfigurator:
    """
    TODO: 2. Buffer reserve for cells array to Config!!
          3. DEFAULT_PARAMS in templates, move out or create different script for cases!
          4. Utils away from the element classes
    """
    def __init__(self):
        self.cases = utils.CaseRef()
        self.utils = utils.Utils()
        self.utils.generate_param()
        self.c_automata = CellularAutomata(self.cases, self.utils)

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

    def configurate_functions_gb(self):
        self.cases.first.microstructure = voronoi.VoronoiMicrostructure(Config.N_CELLS_PER_AXIS)
        self.cases.first.microstructure.generate_voronoi_3d(50, seeds="own")
        self.cases.first.microstructure.show_microstructure(Config.N_CELLS_PER_AXIS)
        self.cases.first.oxidant.microstructure = self.cases.first.microstructure

        self.cases.first.oxidant.diffuse = self.cases.first.oxidant.diffuse_gb
        self.cases.first.active.diffuse = elements.diffuse_bulk_mp

        self.c_automata.get_cur_ioz_bound = self.c_automata.ioz_depth_furthest_inward

        self.c_automata.precip_func = self.c_automata.precipitation_current_case
        self.c_automata.get_combi_ind = self.c_automata.get_combi_ind_standard

        self.c_automata.cases.first_mp.precip_step = precip_step_standard
        self.c_automata.cases.first_mp.check_intersection = ci_single

        self.c_automata.get_cur_dissol_ioz_bound = self.c_automata.ioz_dissolution_where_prod
        self.c_automata.decomposition = self.c_automata.dissolution_test
        self.c_automata.decomposition_intrinsic = self.c_automata.simple_decompose_mp
        self.c_automata.cases.first_mp.decomposition = dissolution_zhou_wei_no_bsf

        self.c_automata.cur_case = self.c_automata.cases.first
        self.c_automata.cur_case_mp = self.c_automata.cases.first_mp

        self.c_automata.cases.first_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                        Config.PRODUCTS.PRIMARY)
        self.c_automata.cases.first_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY)

    def configurate_functions(self):
        self.cases.first.oxidant.diffuse = self.cases.first.oxidant.diffuse_bulk
        self.cases.first.active.diffuse = elements.diffuse_bulk_mp

        self.c_automata.get_cur_ioz_bound = self.c_automata.ioz_depth_from_kinetics
        # self.ca.get_cur_dissol_ioz_bound = self.ca.ioz_dissolution_where_prod

        self.c_automata.precip_func = self.c_automata.precipitation_current_case
        self.c_automata.get_combi_ind = self.c_automata.get_combi_ind_standard

        self.c_automata.cases.first_mp.precip_step = precip_step_standard
        self.c_automata.cases.first_mp.check_intersection = ci_single

        self.c_automata.decomposition = self.c_automata.dissolution_atomic_stop_if_stable
        self.c_automata.decomposition_intrinsic = self.c_automata.simple_decompose_mp
        self.c_automata.cases.first_mp.decomposition = dissolution_zhou_wei_no_bsf

        self.c_automata.cur_case = self.cases.first
        self.c_automata.cur_case_mp = self.cases.first_mp

        self.c_automata.cases.first_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                                Config.PRODUCTS.PRIMARY)
        self.c_automata.cases.first_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY)

    def configurate_functions_K(self):
        self.cases.first.oxidant.diffuse = self.cases.first.oxidant.diffuse_bulk
        self.cases.first.active.diffuse = elements.diffuse_bulk_mp

        self.c_automata.get_cur_ioz_bound = self.c_automata.ioz_depth_furthest_inward
        # self.ca.get_cur_dissol_ioz_bound = self.ca.ioz_dissolution_where_prod

        self.c_automata.precip_func = self.c_automata.precipitation_current_case
        self.c_automata.get_combi_ind = self.c_automata.get_combi_ind_atomic_solub_prod_test

        self.c_automata.cases.first_mp.precip_step = precip_step_standard
        self.c_automata.cases.first_mp.check_intersection = ci_single_no_growth

        # self.c_automata.decomposition = self.c_automata.dissolution_atomic_stop_if_stable
        # self.c_automata.decomposition_intrinsic = self.c_automata.simple_decompose_mp
        # self.c_automata.cases.first_mp.decomposition = dissolution_zhou_wei_no_bsf

        self.c_automata.cur_case = self.cases.first
        self.c_automata.cur_case_mp = self.cases.first_mp

        self.c_automata.cases.first_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                                Config.PRODUCTS.PRIMARY)
        # self.c_automata.cases.first_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY)

    def configurate_functions_cube_dissol(self):
        self.c_automata.get_cur_dissol_ioz_bound = self.c_automata.ioz_dissolution_where_prod

        middle = int(self.c_automata.cells_per_axis / 2)
        side = 20

        self.c_automata.cases.first.product.c3d[middle - side:middle + side, middle - side:middle + side, middle - side:middle + side] = 1
        self.c_automata.cases.first.product.full_c3d[middle - side:middle + side, middle - side:middle + side, middle - side:middle + side] = True
        self.c_automata.cases.first.prod_indexes[middle - side:middle + side] = True


        self.c_automata.decomposition = self.c_automata.dissolution_test
        self.c_automata.decomposition_intrinsic = self.c_automata.simple_decompose_mp
        self.c_automata.cases.first_mp.decomposition = dissolution_zhou_wei_original

        self.c_automata.cur_case = self.c_automata.cases.first
        self.c_automata.cur_case_mp = self.c_automata.cases.first_mp

        self.c_automata.cases.first_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY)

    def configurate_functions_diff_in_scale(self):
        self.cases.first.oxidant.diffuse = self.cases.first.oxidant.diffuse_with_scale_adj

        # scale
        self.cases.first.oxidant.scale = self.cases.first.product.c3d
        self.cases.first.product.c3d[:, :, 30:46] = 1

        self.c_automata.cur_case = self.cases.first
        self.c_automata.cur_case_mp = self.cases.first_mp

    def functions_sec_case(self):
        self.c_automata.primary_oxidant.diffuse = self.c_automata.primary_oxidant.diffuse_bulk
        self.c_automata.primary_active.diffuse = elements.diffuse_bulk_mp
        self.c_automata.secondary_active.diffuse = elements.diffuse_bulk_mp

        self.c_automata.get_cur_ioz_bound = self.c_automata.ioz_depth_from_kinetics

        self.c_automata.precip_func = self.c_automata.precipitation_second_case
        self.c_automata.get_combi_ind = self.c_automata.get_combi_ind_atomic_two_products

        self.c_automata.cases.first_mp.precip_step = precip_step_multi_products
        self.c_automata.cases.first_mp.check_intersection = ci_single

        self.c_automata.cases.second_mp.precip_step = precip_step_multi_products
        self.c_automata.cases.second_mp.check_intersection = ci_single

        self.c_automata.decomposition = self.c_automata.dissolution_atomic_stop_if_stable_two_products
        self.c_automata.decomposition_intrinsic = self.c_automata.simple_decompose_mp

        self.c_automata.cases.first_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL
        self.c_automata.cases.second_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL

        self.c_automata.cases.first_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                                Config.PRODUCTS.PRIMARY)
        self.c_automata.cases.first_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY)

        self.c_automata.cases.second_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.SECONDARY,
                                                                                                 Config.PRODUCTS.SECONDARY)
        self.c_automata.cases.second_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.SECONDARY)

    def configurate_functions_td(self):
        self.c_automata.primary_oxidant.diffuse = self.c_automata.primary_oxidant.diffuse_bulk
        self.c_automata.primary_active.diffuse = elements.diffuse_bulk_mp
        self.c_automata.secondary_active.diffuse = elements.diffuse_bulk_mp

        self.c_automata.precip_func = self.c_automata.precipitation_with_td
        self.c_automata.get_combi_ind = None

        self.c_automata.get_cur_ioz_bound = self.c_automata.ioz_depth_from_kinetics

        self.c_automata.cases.first_mp.precip_step = precip_step_multi_products
        self.c_automata.cases.first_mp.check_intersection = ci_single

        self.c_automata.cases.second_mp.precip_step = precip_step_multi_products
        self.c_automata.cases.second_mp.check_intersection = ci_single

        self.c_automata.decomposition = None
        self.c_automata.decomposition_intrinsic = self.c_automata.simple_decompose_mp

        self.c_automata.cases.first_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL
        self.c_automata.cases.second_mp.decomposition = dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL

        self.c_automata.cases.first_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                                Config.PRODUCTS.PRIMARY)
        self.c_automata.cases.first_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                                  Config.PRODUCTS.PRIMARY)
        self.c_automata.cases.second_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.SECONDARY,
                                                                                                 Config.PRODUCTS.SECONDARY)
        self.c_automata.cases.second_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.SECONDARY,
                                                                                                   Config.PRODUCTS.SECONDARY)

    def configurate_functions_td_all(self):
        self.cases.first.oxidant.diffuse = self.cases.first.oxidant.diffuse_with_scale_adj

        # scale
        self.cases.first.oxidant.scale = self.cases.accumulated_products
        self.cases.first.active.scale = self.cases.accumulated_products

        self.cases.first.active.diffuse = elements.diffuse_bulk_mp
        self.cases.second.active.diffuse = elements.diffuse_bulk_mp

        self.c_automata.precip_func = self.c_automata.precipitation_with_td
        self.c_automata.get_combi_ind = None

        self.c_automata.get_cur_ioz_bound = self.c_automata.ioz_depth_furthest_inward

        self.cases.first_mp.precip_step = precip_step_multi_products
        self.cases.first_mp.check_intersection = ci_multi

        self.cases.second_mp.precip_step = precip_step_multi_products
        self.cases.second_mp.check_intersection = ci_multi

        self.cases.third_mp.precip_step = precip_step_multi_products
        self.cases.third_mp.check_intersection = ci_multi

        self.cases.fourth_mp.precip_step = precip_step_multi_products
        self.cases.fourth_mp.check_intersection = ci_multi

        self.cases.fifth_mp.precip_step = precip_step_multi_products
        self.cases.fifth_mp.check_intersection = ci_multi_no_active

        self.c_automata.decomposition = None
        self.c_automata.decomposition_intrinsic = self.c_automata.simple_decompose_mp

        self.cases.first_mp.decomposition = dissolution_zhou_wei_no_bsf
        self.cases.second_mp.decomposition = dissolution_zhou_wei_no_bsf
        self.cases.third_mp.decomposition = dissolution_zhou_wei_no_bsf
        self.cases.fourth_mp.decomposition = dissolution_zhou_wei_no_bsf
        self.cases.fifth_mp.decomposition = dissolution_zhou_wei_no_bsf

        self.cases.first_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY, Config.PRODUCTS.PRIMARY)
        self.cases.first_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY)
        self.cases.second_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.SECONDARY, Config.PRODUCTS.SECONDARY)
        self.cases.second_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.SECONDARY)
        self.cases.third_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY, Config.PRODUCTS.TERNARY)
        self.cases.third_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY)
        self.cases.fourth_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY, Config.PRODUCTS.QUATERNARY)
        self.cases.fourth_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY)
        self.cases.fifth_mp.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY, Config.PRODUCTS.QUINT)
        self.cases.fifth_mp.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY)

    def run_simulation(self):
        self.begin = time.time()
        for self.c_automata.iteration in progressbar.progressbar(range(Config.N_ITERATIONS)):
            if keyboard.is_pressed('ctrl+g+m'):
                break
            self.c_automata.precip_func()
            # self.c_automata.decomposition()
            self.c_automata.diffusion_inward()
            self.c_automata.diffusion_outward()
            # self.calc_precipitation_front_only_cells()
            # self.diffusion_outward_with_mult_srtide()
            # self.save_results_only_prod_prime()
            # self.save_results_only_prod()

        end = time.time()
        self.elapsed_time = (end - self.begin)
        self.db.insert_time(self.elapsed_time)
        self.db.conn.commit()

    def init_inward(self):
        # self.ca.primary_oxidant = elements.OxidantElem(Config.OXIDANTS.PRIMARY, self.utils)
        # self.ca.cases.first.oxidant = self.ca.primary_oxidant
        self.cases.first.oxidant = elements.OxidantElem(Config.OXIDANTS.PRIMARY, self.utils)

        # self.ca.cases.second.oxidant = self.ca.primary_oxidant
        self.cases.second.oxidant = self.cases.first.oxidant

        self.cases.first_mp.oxidant_c3d_shm_mdata = self.cases.first.oxidant.c3d_shm_mdata
        self.cases.second_mp.oxidant_c3d_shm_mdata = self.cases.first.oxidant.c3d_shm_mdata

        self.cases.third.oxidant = self.cases.first.oxidant
        self.cases.fourth.oxidant = self.cases.first.oxidant
        self.cases.fifth.oxidant = self.cases.first.oxidant

        self.cases.third_mp.oxidant_c3d_shm_mdata = self.cases.first.oxidant.c3d_shm_mdata
        self.cases.fourth_mp.oxidant_c3d_shm_mdata = self.cases.first.oxidant.c3d_shm_mdata
        self.cases.fifth_mp.oxidant_c3d_shm_mdata = self.cases.first.oxidant.c3d_shm_mdata

        # ---------------------------------------------------
        if Config.OXIDANTS.SECONDARY_EXISTENCE:
            # self.ca.secondary_oxidant = elements.OxidantElem(Config.OXIDANTS.SECONDARY, self.utils)
            self.cases.third.oxidant = elements.OxidantElem(Config.OXIDANTS.SECONDARY, self.utils)

            # self.cases.third.oxidant = self.ca.secondary_oxidant
            self.cases.fourth.oxidant = self.cases.third.oxidant

            self.cases.third_mp.oxidant_c3d_shm_mdata = self.cases.third.oxidant.c3d_shm_mdata
            self.cases.fourth_mp.oxidant_c3d_shm_mdata = self.cases.third.oxidant.c3d_shm_mdata

    def init_outward(self):
        # self.ca.primary_active = elements.ActiveElem(Config.ACTIVES.PRIMARY)
        self.cases.first.active = elements.ActiveElem(Config.ACTIVES.PRIMARY)
        self.cases.third.active = self.cases.first.active
        # ---------------------------------------------------
        # c3d
        self.cases.first_mp.active_c3d_shm_mdata = self.cases.first.active.c3d_shm_mdata
        self.cases.third_mp.active_c3d_shm_mdata = self.cases.first.active.c3d_shm_mdata

        self.cases.fifth_mp.active_c3d_shm_mdata = self.cases.first.active.c3d_shm_mdata  # JUST FOR SHAPE!!!

        # cells
        self.cases.first_mp.active_cells_shm_mdata = self.cases.first.active.cells_shm_mdata
        self.cases.third_mp.active_cells_shm_mdata = self.cases.first.active.cells_shm_mdata
        # dirs
        self.cases.first_mp.active_dirs_shm_mdata = self.cases.first.active.dirs_shm_mdata
        self.cases.third_mp.active_dirs_shm_mdata = self.cases.first.active.dirs_shm_mdata

        # ---------------------------------------------------
        if Config.ACTIVES.SECONDARY_EXISTENCE:
            # self.secondary_active = elements.ActiveElem(Config.ACTIVES.SECONDARY)
            self.cases.second.active = elements.ActiveElem(Config.ACTIVES.SECONDARY)
            self.cases.fourth.active = self.cases.second.active
            # ---------------------------------------------------
            # c3d
            self.cases.second_mp.active_c3d_shm_mdata = self.cases.second.active.c3d_shm_mdata
            self.cases.fourth_mp.active_c3d_shm_mdata = self.cases.second.active.c3d_shm_mdata
            # cells
            self.cases.second_mp.active_cells_shm_mdata = self.cases.second.active.cells_shm_mdata
            self.cases.fourth_mp.active_cells_shm_mdata = self.cases.second.active.cells_shm_mdata
            # dirs
            self.cases.second_mp.active_dirs_shm_mdata = self.cases.second.active.dirs_shm_mdata
            self.cases.fourth_mp.active_dirs_shm_mdata = self.cases.second.active.dirs_shm_mdata

    def init_product(self):
        # c3d_init
        tmp = np.zeros((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), dtype=np.ubyte)
        self.cases.precip_3d_init_shm = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.cases.precip_3d_init = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.cases.precip_3d_init_shm.buf)
        np.copyto(self.cases.precip_3d_init, tmp)
        self.cases.precip_3d_init_shm_mdata = SharedMetaData(self.cases.precip_3d_init_shm.name, tmp.shape, tmp.dtype)

        # accumulated products
        tmp = np.zeros((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), dtype=np.ubyte)
        self.cases.accumulated_products_shm = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        self.cases.accumulated_products = np.ndarray(tmp.shape, dtype=tmp.dtype,
                                                     buffer=self.cases.accumulated_products_shm.buf)
        np.copyto(self.cases.accumulated_products, tmp)
        self.cases.accumulated_products_shm_mdata = SharedMetaData(self.cases.accumulated_products_shm.name, tmp.shape,
                                                                   tmp.dtype)

        if Config.ACTIVES.SECONDARY_EXISTENCE and Config.OXIDANTS.SECONDARY_EXISTENCE:
            print("no implementation")
            # self.init_first_case()
            # self.init_second_case()
            # self.init_third_case()
            # self.init_fourth_case()
            # self.init_fifth_case()

        elif Config.ACTIVES.SECONDARY_EXISTENCE and not Config.OXIDANTS.SECONDARY_EXISTENCE:
            self.init_case(self.cases.first, self.cases.first_mp, Config.PRODUCTS.PRIMARY)
            self.init_case(self.cases.second, self.cases.second_mp, Config.PRODUCTS.SECONDARY)
            self.init_case(self.cases.third, self.cases.third_mp, Config.PRODUCTS.TERNARY)
            self.init_case(self.cases.fourth, self.cases.fourth_mp, Config.PRODUCTS.QUATERNARY)
            self.init_case(self.cases.fifth, self.cases.fifth_mp, Config.PRODUCTS.QUINT)
        else:
            self.init_case(self.cases.first, self.cases.first_mp, Config.PRODUCTS.PRIMARY)

    def init_case(self, case, case_mp, product_config):
        case_mp.cells_per_axis = Config.N_CELLS_PER_AXIS
        # prod init
        case.product = elements.Product(product_config)
        case_mp.oxidation_number = case.product.oxidation_number
        case_mp.threshold_inward = product_config.THRESHOLD_INWARD
        case_mp.threshold_outward = product_config.THRESHOLD_OUTWARD

        # to check with
        case_mp.to_check_with_shm_mdata = self.cases.accumulated_products_shm_mdata

        # c3d
        case_mp.product_c3d_shm_mdata = case.product.c3d_shm_mdata

        # full_c3d
        case_mp.full_shm_mdata = case.product.full_shm_mdata

        # product indexes
        tmp = np.full(Config.N_CELLS_PER_AXIS, False, dtype=bool)
        case.shm_pool["product_indexes"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        case.prod_indexes = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=case.shm_pool["product_indexes"].buf)
        np.copyto(case.prod_indexes, tmp)
        case_mp.prod_indexes_shm_mdata = SharedMetaData(case.shm_pool["product_indexes"].name, tmp.shape, tmp.dtype)

        # ind not stable
        tmp = np.full(Config.N_CELLS_PER_AXIS, True, dtype=bool)
        case.shm_pool["product_ind_not_stab"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
        case.product_ind_not_stab = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=case.shm_pool["product_ind_not_stab"].buf)
        np.copyto(case.product_ind_not_stab, tmp)
        case_mp.prod_indexes_not_stab_shm_mdata = SharedMetaData(case.shm_pool["product_ind_not_stab"].name, tmp.shape, tmp.dtype)

        # functions
        case_mp.go_around_func_ref = go_around_mult_oxid_n_BOOl
        case.fix_init_precip_func_ref = self.c_automata.fix_init_precip_int

        # c3d_init
        case.precip_3d_init = self.cases.precip_3d_init
        case_mp.precip_3d_init_shm_mdata = self.cases.precip_3d_init_shm_mdata

        # fix full cells
        case_mp.fix_full_cells = elements.fix_full_cells

    def save_results(self):
        if Config.STRIDE > Config.N_ITERATIONS:
            self.cases.first.active.transform_to_descards()
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.cases.second.active.transform_to_descards()
        if Config.INWARD_DIFFUSION:
            self.db.insert_particle_data("primary_oxidant", self.c_automata.iteration, self.cases.first.oxidant.cells)
            if Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_oxidant", self.c_automata.iteration, self.cases.second.oxidant.cells)
        if Config.OUTWARD_DIFFUSION:
            self.db.insert_particle_data("primary_active", self.c_automata.iteration, self.cases.first.active.get_cells_coords())
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_active", self.c_automata.iteration, self.cases.second.active.get_cells_coords())
        if Config.COMPUTE_PRECIPITATION:
            self.db.insert_particle_data("primary_product", self.c_automata.iteration, self.cases.first.product.transform_c3d())
            if Config.ACTIVES.SECONDARY_EXISTENCE and Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_product", self.c_automata.iteration, self.cases.second.product.transform_c3d())
                self.db.insert_particle_data("ternary_product", self.c_automata.iteration, self.cases.third.product.transform_c3d())
                self.db.insert_particle_data("quaternary_product", self.c_automata.iteration, self.cases.fourth.product.transform_c3d())
            elif Config.ACTIVES.SECONDARY_EXISTENCE and not Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_product", self.c_automata.iteration, self.cases.second.product.transform_c3d())
        if Config.STRIDE > Config.N_ITERATIONS:
            self.cases.first.active.transform_to_3d(self.c_automata.curr_max_furthest)
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.cases.second.active.transform_to_3d(self.c_automata.curr_max_furthest)

    def save_results_custom(self):
        if Config.STRIDE > Config.N_ITERATIONS:
            self.cases.first.active.transform_to_descards()
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.cases.second.active.transform_to_descards()

        self.db.insert_particle_data("primary_oxidant", self.c_automata.iteration, self.cases.first.oxidant.cells)

        self.db.insert_particle_data("primary_active", self.c_automata.iteration, self.cases.first.active.get_cells_coords())
        self.db.insert_particle_data("secondary_active", self.c_automata.iteration, self.cases.second.active.get_cells_coords())

        self.db.insert_particle_data("primary_product", self.c_automata.iteration, self.cases.first.product.transform_c3d())
        self.db.insert_particle_data("secondary_product", self.c_automata.iteration, self.cases.second.product.transform_c3d())
        self.db.insert_particle_data("ternary_product", self.c_automata.iteration, self.cases.third.product.transform_c3d())
        self.db.insert_particle_data("quaternary_product", self.c_automata.iteration, self.cases.fourth.product.transform_c3d())
        self.db.insert_particle_data("quint_product", self.c_automata.iteration, self.cases.fifth.product.transform_c3d())

        if Config.STRIDE > Config.N_ITERATIONS:
            self.cases.first.active.transform_to_3d(self.c_automata.curr_max_furthest)
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.cases.second.active.transform_to_3d(self.c_automata.curr_max_furthest)

    def calc_precipitation_front_only_cells(self):
        """
        Calculating a position of a precipitation front, considering only cells concentrations without any scaling!
        As a boundary a product fraction of 0,1% is used.
        """
        product = np.array([np.sum(self.c_automata.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.c_automata.cells_per_axis)], dtype=np.uint32)
        product = product / (self.c_automata.cells_per_axis ** 2)
        threshold = Config.ACTIVES.PRIMARY.CELLS_CONCENTRATION
        for rev_index, precip_conc in enumerate(np.flip(product)):
            if precip_conc > threshold / 100:
                position = (len(product) - 1 - rev_index) * Config.SIZE * 10 ** 6 \
                           / self.c_automata.cells_per_axis
                sqr_time = ((self.c_automata.iteration + 1) * Config.SIM_TIME / (self.c_automata.n_iter * 3600)) ** (1 / 2)
                self.db.insert_precipitation_front(sqr_time, position, "p")
                break

    def terminate_workers(self):
        self.c_automata.pool.close()
        self.c_automata.pool.join()
        print("TERMINATED PROPERLY!")

    def unlink(self):
        self.cases.close_shms()
        print("UNLINKED PROPERLY!")

    def save_results_only_prod_prime(self):
        self.db.insert_particle_data("primary_product", self.c_automata.iteration, self.cases.first.product.transform_c3d())

    def save_results_only_prod(self):
        self.db.insert_particle_data("primary_product", self.c_automata.iteration, self.cases.first.product.transform_c3d())
        self.db.insert_particle_data("secondary_product", self.c_automata.iteration, self.cases.second.product.transform_c3d())
        self.db.insert_particle_data("ternary_product", self.c_automata.iteration, self.cases.third.product.transform_c3d())
        self.db.insert_particle_data("quaternary_product", self.c_automata.iteration, self.cases.fourth.product.transform_c3d())
        self.db.insert_particle_data("quint_product", self.c_automata.iteration, self.cases.fifth.product.transform_c3d())

    def save_results_only_prod_secondary(self):
        self.db.insert_particle_data("secondary_product", self.c_automata.iteration, self.cases.second.product.transform_c3d())

    def save_results_prod_and_inw(self):
        self.db.insert_particle_data("primary_product", self.c_automata.iteration, self.cases.first.product.transform_c3d())
        self.db.insert_particle_data("primary_oxidant", self.c_automata.iteration, self.cases.first.oxidant.cells)

    def save_results_only_inw(self):
        self.db.insert_particle_data("primary_oxidant", self.c_automata.iteration, self.cases.first.oxidant.cells)

    def insert_last_it(self):
        self.db.insert_last_iteration(self.c_automata.iteration)

    #
    # def init_first_case(self):
    #     self.cases.first_mp.cells_per_axis = Config.N_CELLS_PER_AXIS
    #     self.cases.first.product = elements.Product(Config.PRODUCTS.PRIMARY)
    #
    #     self.cases.first_mp.oxidation_number = self.cases.first.product.oxidation_number
    #
    #     self.cases.first_mp.threshold_inward = Config.PRODUCTS.PRIMARY.THRESHOLD_INWARD
    #     self.cases.first_mp.threshold_outward = Config.PRODUCTS.PRIMARY.THRESHOLD_OUTWARD
    #
    #     # to check with
    #     self.cases.first_mp.to_check_with_shm_mdata = self.cases.accumulated_products_shm_mdata
    #
    #     # c3d
    #     self.cases.first_mp.product_c3d_shm_mdata = self.cases.first.product.c3d_shm_mdata
    #
    #     # full_c3d
    #     self.cases.first_mp.full_shm_mdata = self.cases.first.product.full_shm_mdata
    #
    #     # product indexes
    #     tmp = np.full(Config.N_CELLS_PER_AXIS, False, dtype=bool)
    #     self.cases.first.shm_pool["product_indexes"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.cases.first.prod_indexes = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.cases.first.shm_pool["product_indexes"].buf)
    #     np.copyto(self.cases.first.prod_indexes, tmp)
    #     self.cases.first_mp.prod_indexes_shm_mdata = SharedMetaData(self.cases.first.shm_pool["product_indexes"].name, tmp.shape, tmp.dtype)
    #
    #     # ind not stable
    #     tmp = np.full(Config.N_CELLS_PER_AXIS, True, dtype=bool)
    #     self.cases.first.shm_pool["product_ind_not_stab"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.cases.first.product_ind_not_stab = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.cases.first.shm_pool["product_ind_not_stab"].buf)
    #     np.copyto(self.cases.first.product_ind_not_stab, tmp)
    #     self.cases.first_mp.prod_indexes_not_stab_shm_mdata = SharedMetaData(self.cases.first.shm_pool["product_ind_not_stab"].name, tmp.shape, tmp.dtype)
    #
    #     # functions
    #     if self.cases.first.product.oxidation_number == 1:
    #         self.cases.first_mp.go_around_func_ref = go_around_mult_oxid_n_BOOl
    #         self.cases.first.fix_init_precip_func_ref = self.c_automata.fix_init_precip_bool
    #         my_type = bool
    #     else:
    #         self.cases.first_mp.go_around_func_ref = go_around_mult_oxid_n_BOOl
    #         self.cases.first.fix_init_precip_func_ref = self.c_automata.fix_init_precip_int
    #         my_type = np.ubyte
    #
    #     # c3d_init
    #     tmp = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), False, dtype=my_type)
    #     self.cases.first.shm_pool["precip_3d_init"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.cases.first.precip_3d_init = np.ndarray(tmp.shape, dtype=tmp.dtype, buffer=self.cases.first.shm_pool["precip_3d_init"].buf)
    #     np.copyto(self.cases.first.precip_3d_init, tmp)
    #     self.cases.first_mp.precip_3d_init_shm_mdata = SharedMetaData(self.cases.first.shm_pool["precip_3d_init"].name,
    #                                                                              tmp.shape, tmp.dtype)
    #
    #     # fix full cells
    #     self.cases.first_mp.fix_full_cells = elements.fix_full_cells
    #
    # def init_second_case(self):
    #     self.c_automata.cases.second_mp.cells_per_axis = Config.N_CELLS_PER_AXIS
    #
    #     self.cases.second.product = elements.Product(Config.PRODUCTS.SECONDARY)
    #     self.cases.second_mp.oxidation_number = self.c_automata.cases.second.product.oxidation_number
    #
    #     self.cases.second_mp.threshold_inward = Config.PRODUCTS.SECONDARY.THRESHOLD_INWARD
    #     self.cases.second_mp.threshold_outward = Config.PRODUCTS.SECONDARY.THRESHOLD_OUTWARD
    #
    #     # to check with
    #     self.cases.second_mp.to_check_with_shm_mdata = self.cases.accumulated_products_shm_mdata
    #
    #     # c3d
    #     self.cases.second_mp.product_c3d_shm_mdata = self.cases.second.product.c3d_shm_mdata
    #     # full_c3d
    #     self.c_automata.cases.second_mp.full_shm_mdata = self.cases.second.product.full_shm_mdata
    #     # product indexes
    #     tmp = np.full(Config.N_CELLS_PER_AXIS, False, dtype=bool)
    #     self.c_automata.cases.second.shm_pool["product_indexes"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.c_automata.cases.second.prod_indexes = np.ndarray(tmp.shape, dtype=tmp.dtype,
    #                                                            buffer=self.c_automata.cases.second.shm_pool["product_indexes"].buf)
    #     np.copyto(self.c_automata.cases.second.prod_indexes, tmp)
    #     self.c_automata.cases.second_mp.prod_indexes_shm_mdata = SharedMetaData(self.c_automata.cases.second.shm_pool["product_indexes"].name,
    #                                                                             tmp.shape, tmp.dtype)
    #     # ind not stable
    #     tmp = np.full(Config.N_CELLS_PER_AXIS, True, dtype=bool)
    #     self.c_automata.cases.second.shm_pool["product_ind_not_stab"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.c_automata.cases.second.product_ind_not_stab = np.ndarray(tmp.shape, dtype=tmp.dtype,
    #                                                                    buffer=self.c_automata.cases.second.shm_pool["product_ind_not_stab"].buf)
    #     np.copyto(self.c_automata.cases.second.product_ind_not_stab, tmp)
    #     self.c_automata.cases.second_mp.prod_indexes_not_stab_shm_mdata = SharedMetaData(self.c_automata.cases.second.shm_pool["product_ind_not_stab"].name,
    #                                                                                      tmp.shape, tmp.dtype)
    #     # c3d_init
    #     if self.c_automata.cases.second.product.oxidation_number == 1:
    #         self.c_automata.cases.second_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
    #         self.c_automata.cases.second.fix_init_precip_func_ref = self.c_automata.fix_init_precip_bool
    #         my_type = bool
    #     else:
    #         self.c_automata.cases.second_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
    #         self.c_automata.cases.second.fix_init_precip_func_ref = self.c_automata.fix_init_precip_int
    #         my_type = np.ubyte
    #
    #     tmp = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), False,
    #                   dtype=my_type)
    #     self.c_automata.cases.second.shm_pool["precip_3d_init"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.c_automata.cases.second.precip_3d_init = np.ndarray(tmp.shape, dtype=tmp.dtype,
    #                                                              buffer=self.c_automata.cases.second.shm_pool["precip_3d_init"].buf)
    #     np.copyto(self.c_automata.cases.second.precip_3d_init, tmp)
    #     self.c_automata.cases.second_mp.precip_3d_init_shm_mdata = SharedMetaData(self.c_automata.cases.second.shm_pool["precip_3d_init"].name,
    #                                                                               tmp.shape, tmp.dtype)
    #     # fix full cells
    #     self.c_automata.cases.second_mp.fix_full_cells = elements.fix_full_cells
    #
    # def init_third_case(self):
    #     self.c_automata.cases.third_mp.cells_per_axis = self.c_automata.cells_per_axis
    #     self.c_automata.ternary_product = elements.Product(Config.PRODUCTS.TERNARY)
    #     self.c_automata.cases.third.product = self.c_automata.ternary_product
    #     self.c_automata.cases.third_mp.oxidation_number = self.c_automata.ternary_product.oxidation_number
    #
    #     self.c_automata.cases.third_mp.threshold_inward = Config.PRODUCTS.TERNARY.THRESHOLD_INWARD
    #     self.c_automata.cases.third_mp.threshold_outward = Config.PRODUCTS.TERNARY.THRESHOLD_OUTWARD
    #
    #     # to check with
    #     self.c_automata.cases.third_mp.to_check_with_shm_mdata = self.c_automata.cases.accumulated_products_shm_mdata
    #
    #     # c3d
    #     self.c_automata.cases.third_mp.product_c3d_shm_mdata = self.c_automata.ternary_product.c3d_shm_mdata
    #     # full_c3d
    #     self.c_automata.cases.third_mp.full_shm_mdata = self.c_automata.ternary_product.full_shm_mdata
    #     # product indexes
    #     tmp = np.full(Config.N_CELLS_PER_AXIS, False, dtype=bool)
    #     self.c_automata.cases.third.shm_pool["product_indexes"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.c_automata.cases.third.prod_indexes = np.ndarray(tmp.shape, dtype=tmp.dtype,
    #                                                           buffer=self.c_automata.cases.third.shm_pool["product_indexes"].buf)
    #     np.copyto(self.c_automata.cases.third.prod_indexes, tmp)
    #     self.c_automata.cases.third_mp.prod_indexes_shm_mdata = SharedMetaData(
    #         self.c_automata.cases.third.shm_pool["product_indexes"].name,
    #         tmp.shape, tmp.dtype)
    #     # ind not stable
    #     tmp = np.full(Config.N_CELLS_PER_AXIS, True, dtype=bool)
    #     self.c_automata.cases.third.shm_pool["product_ind_not_stab"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.c_automata.cases.third.product_ind_not_stab = np.ndarray(tmp.shape, dtype=tmp.dtype,
    #                                                                   buffer=self.c_automata.cases.third.shm_pool[
    #                                                               "product_ind_not_stab"].buf)
    #     np.copyto(self.c_automata.cases.third.product_ind_not_stab, tmp)
    #     self.c_automata.cases.third_mp.prod_indexes_not_stab_shm_mdata = SharedMetaData(
    #         self.c_automata.cases.third.shm_pool["product_ind_not_stab"].name,
    #         tmp.shape, tmp.dtype)
    #     # c3d_init
    #     if self.c_automata.cases.third.product.oxidation_number == 1:
    #         self.c_automata.cases.third_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
    #         self.c_automata.cases.third.fix_init_precip_func_ref = self.c_automata.fix_init_precip_bool
    #         my_type = bool
    #     else:
    #         self.c_automata.cases.third_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
    #         self.c_automata.cases.third.fix_init_precip_func_ref = self.c_automata.fix_init_precip_int
    #         my_type = np.ubyte
    #
    #     tmp = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), False,
    #                   dtype=my_type)
    #     self.c_automata.cases.third.shm_pool["precip_3d_init"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.c_automata.cases.third.precip_3d_init = np.ndarray(tmp.shape, dtype=tmp.dtype,
    #                                                             buffer=self.c_automata.cases.third.shm_pool["precip_3d_init"].buf)
    #     np.copyto(self.c_automata.cases.third.precip_3d_init, tmp)
    #     self.c_automata.cases.third_mp.precip_3d_init_shm_mdata = SharedMetaData(
    #         self.c_automata.cases.third.shm_pool["precip_3d_init"].name,
    #         tmp.shape, tmp.dtype)
    #     # fix full cells
    #     self.c_automata.cases.third_mp.fix_full_cells = elements.fix_full_cells
    #
    # def init_fourth_case(self):
    #     self.c_automata.cases.fourth_mp.cells_per_axis = self.c_automata.cells_per_axis
    #     self.c_automata.quaternary_product = elements.Product(Config.PRODUCTS.QUATERNARY)
    #     self.c_automata.cases.fourth.product = self.c_automata.quaternary_product
    #     self.c_automata.cases.fourth_mp.oxidation_number = self.c_automata.quaternary_product.oxidation_number
    #
    #     self.c_automata.cases.fourth_mp.threshold_inward = Config.PRODUCTS.QUATERNARY.THRESHOLD_INWARD
    #     self.c_automata.cases.fourth_mp.threshold_outward = Config.PRODUCTS.QUATERNARY.THRESHOLD_OUTWARD
    #
    #     # to check with
    #     self.c_automata.cases.fourth_mp.to_check_with_shm_mdata = self.c_automata.cases.accumulated_products_shm_mdata
    #
    #     # c3d
    #     self.c_automata.cases.fourth_mp.product_c3d_shm_mdata = self.c_automata.quaternary_product.c3d_shm_mdata
    #     # full_c3d
    #     self.c_automata.cases.fourth_mp.full_shm_mdata = self.c_automata.quaternary_product.full_shm_mdata
    #     # product indexes
    #     tmp = np.full(Config.N_CELLS_PER_AXIS, False, dtype=bool)
    #     self.c_automata.cases.fourth.shm_pool["product_indexes"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.c_automata.cases.fourth.prod_indexes = np.ndarray(tmp.shape, dtype=tmp.dtype,
    #                                                            buffer=self.c_automata.cases.fourth.shm_pool["product_indexes"].buf)
    #     np.copyto(self.c_automata.cases.fourth.prod_indexes, tmp)
    #     self.c_automata.cases.fourth_mp.prod_indexes_shm_mdata = SharedMetaData(
    #         self.c_automata.cases.fourth.shm_pool["product_indexes"].name,
    #         tmp.shape, tmp.dtype)
    #     # ind not stable
    #     tmp = np.full(Config.N_CELLS_PER_AXIS, True, dtype=bool)
    #     self.c_automata.cases.fourth.shm_pool["product_ind_not_stab"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.c_automata.cases.fourth.product_ind_not_stab = np.ndarray(tmp.shape, dtype=tmp.dtype,
    #                                                                    buffer=self.c_automata.cases.fourth.shm_pool[
    #                                                                "product_ind_not_stab"].buf)
    #     np.copyto(self.c_automata.cases.fourth.product_ind_not_stab, tmp)
    #     self.c_automata.cases.fourth_mp.prod_indexes_not_stab_shm_mdata = SharedMetaData(
    #         self.c_automata.cases.fourth.shm_pool["product_ind_not_stab"].name,
    #         tmp.shape, tmp.dtype)
    #     # c3d_init
    #     if self.c_automata.cases.fourth.product.oxidation_number == 1:
    #         self.c_automata.cases.fourth_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
    #         self.c_automata.cases.fourth.fix_init_precip_func_ref = self.c_automata.fix_init_precip_bool
    #         my_type = bool
    #     else:
    #         self.c_automata.cases.fourth_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
    #         self.c_automata.cases.fourth.fix_init_precip_func_ref = self.c_automata.fix_init_precip_int
    #         my_type = np.ubyte
    #
    #     tmp = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), False,
    #                   dtype=my_type)
    #     self.c_automata.cases.fourth.shm_pool["precip_3d_init"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.c_automata.cases.fourth.precip_3d_init = np.ndarray(tmp.shape, dtype=tmp.dtype,
    #                                                              buffer=self.c_automata.cases.fourth.shm_pool["precip_3d_init"].buf)
    #     np.copyto(self.c_automata.cases.fourth.precip_3d_init, tmp)
    #     self.c_automata.cases.fourth_mp.precip_3d_init_shm_mdata = SharedMetaData(
    #         self.c_automata.cases.fourth.shm_pool["precip_3d_init"].name,
    #         tmp.shape, tmp.dtype)
    #     # fix full cells
    #     self.c_automata.cases.fourth_mp.fix_full_cells = elements.fix_full_cells
    #
    # def init_fifth_case(self):
    #     self.c_automata.cases.fifth_mp.cells_per_axis = self.c_automata.cells_per_axis
    #     self.c_automata.quint_product = elements.Product(Config.PRODUCTS.QUINT)
    #     self.c_automata.cases.fifth.product = self.c_automata.quint_product
    #     self.c_automata.cases.fifth_mp.oxidation_number = self.c_automata.quint_product.oxidation_number
    #
    #     self.c_automata.cases.fifth_mp.threshold_inward = Config.PRODUCTS.QUINT.THRESHOLD_INWARD
    #     self.c_automata.cases.fifth_mp.threshold_outward = Config.PRODUCTS.QUINT.THRESHOLD_OUTWARD
    #
    #     # to check with
    #     self.c_automata.cases.fifth_mp.to_check_with_shm_mdata = self.c_automata.cases.accumulated_products_shm_mdata
    #
    #     # c3d
    #     self.c_automata.cases.fifth_mp.product_c3d_shm_mdata = self.c_automata.quint_product.c3d_shm_mdata
    #     # full_c3d
    #     self.c_automata.cases.fifth_mp.full_shm_mdata = self.c_automata.quint_product.full_shm_mdata
    #     # product indexes
    #     tmp = np.full(Config.N_CELLS_PER_AXIS, False, dtype=bool)
    #     self.c_automata.cases.fifth.shm_pool["product_indexes"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.c_automata.cases.fifth.prod_indexes = np.ndarray(tmp.shape, dtype=tmp.dtype,
    #                                                           buffer=self.c_automata.cases.fifth.shm_pool["product_indexes"].buf)
    #     np.copyto(self.c_automata.cases.fifth.prod_indexes, tmp)
    #     self.c_automata.cases.fifth_mp.prod_indexes_shm_mdata = SharedMetaData(
    #         self.c_automata.cases.fifth.shm_pool["product_indexes"].name,
    #         tmp.shape, tmp.dtype)
    #     # ind not stable
    #     tmp = np.full(Config.N_CELLS_PER_AXIS, True, dtype=bool)
    #     self.c_automata.cases.fifth.shm_pool["product_ind_not_stab"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.c_automata.cases.fifth.product_ind_not_stab = np.ndarray(tmp.shape, dtype=tmp.dtype,
    #                                                                   buffer=self.c_automata.cases.fifth.shm_pool[
    #                                                               "product_ind_not_stab"].buf)
    #     np.copyto(self.c_automata.cases.fifth.product_ind_not_stab, tmp)
    #     self.c_automata.cases.fifth_mp.prod_indexes_not_stab_shm_mdata = SharedMetaData(
    #         self.c_automata.cases.fifth.shm_pool["product_ind_not_stab"].name,
    #         tmp.shape, tmp.dtype)
    #     # c3d_init
    #     if self.c_automata.cases.fifth.product.oxidation_number == 1:
    #         self.c_automata.cases.fifth_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
    #         self.c_automata.cases.fifth.fix_init_precip_func_ref = self.c_automata.fix_init_precip_bool
    #         my_type = bool
    #     else:
    #         self.c_automata.cases.fifth_mp.go_around_func_ref = go_around_mult_oxid_n_also_partial_neigh_aip_MP
    #         self.c_automata.cases.fifth.fix_init_precip_func_ref = self.c_automata.fix_init_precip_int
    #         my_type = np.ubyte
    #
    #     tmp = np.full((Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS, Config.N_CELLS_PER_AXIS + 1), False,
    #                   dtype=my_type)
    #     self.c_automata.cases.fifth.shm_pool["precip_3d_init"] = shared_memory.SharedMemory(create=True, size=tmp.nbytes)
    #     self.c_automata.cases.fifth.precip_3d_init = np.ndarray(tmp.shape, dtype=tmp.dtype,
    #                                                             buffer=self.c_automata.cases.fifth.shm_pool["precip_3d_init"].buf)
    #     np.copyto(self.c_automata.cases.fifth.precip_3d_init, tmp)
    #     self.c_automata.cases.fifth_mp.precip_3d_init_shm_mdata = SharedMetaData(
    #         self.c_automata.cases.fifth.shm_pool["precip_3d_init"].name,
    #         tmp.shape, tmp.dtype)
    #     # fix full cells
    #     self.c_automata.cases.fifth_mp.fix_full_cells = elements.fix_full_cells