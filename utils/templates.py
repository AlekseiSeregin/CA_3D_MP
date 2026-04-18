import numpy as np
from configuration import Config


class CaseSetUp:
    def __init__(self):
        self.is_active = False
        self.oxidant = None
        self.active = None
        self.microstructure = None
        self.prod_indexes = None
        self.product_ind_not_stab = None
        self.dissolution_probabilities = None
        self.shm_pool = {"product_indexes": None,
                         "product_ind_not_stab": None,
                         "precip_3d_init": None}

    def close_and_unlink_shared_memory(self):
        for key, shm in self.shm_pool.items():
            if shm is not None:
                shm.close()
                shm.unlink()

        if self.oxidant is not None:
            self.oxidant.close_and_unlink_shm()

        if self.active is not None:
            self.active.close_and_unlink_shm()


class CaseSetUpMP:
    def __init__(self):
        self.is_active = False
        self.active_c3d_shm_mdata = None
        self.active_cells_shm_mdata = None
        self.active_dirs_shm_mdata = None

        self.oxidant_c3d_shm_mdata = None
        self.product_c3d_shm_mdata = None
        self.oxidation_number = None
  
        self.prod_indexes_shm_mdata = None
        self.prod_indexes_not_stab_shm_mdata = None
        self.product_state_shm_mdata = None
        self.product_phase_id = 0

        self.go_around_func_ref = None
        self.precip_3d_init_shm_mdata = None

        self.nucleation_probabilities = None
        self.dissolution_probabilities = None

        self.precip_step = None
        self.check_intersection = None

        self.decomposition = None
        self.fix_full_cells = None

        self.threshold_inward = None
        self.threshold_outward = None

        self.cells_per_axis = Config.N_CELLS_PER_AXIS

        self.nucleation_mode = Config.NUCLEATION_MODE
        self.nucleation_kernel_runner = None
        self.product_key = None
        self.product_cfg = None
        self.product_element = None
        self.product_components = ()
        self.stage_priority = 0
        self.jm_identifier = ""
        self.outward_element = ""
        self.stoich_frac_items = ()

        self.plane_indexes = []
        self.dissolution_plane_indexes = []
        # JMatPro block mode: uint16 (bx, by) with z bit bz — where CA fraction exceeds equilibrium (dissolve).
        self.dissolution_block_mask_bits = None


class CaseRef:
    def __init__(self):
        # Canonical product-stage containers (legacy first/second/... names remain aliases).
        self.product_cases = []
        self.product_cases_mp = []

        self.product_state = None
        self.product_state_shm = None
        self.product_state_shm_mdata = None

        self.precip_3d_init = None
        self.precip_3d_init_shm = None
        self.precip_3d_init_shm_mdata = None

        self.all_oxidants = []
        self.all_actives = []
        self.all_products = []
        self.all_elements = []

        self.all_cases = self.product_cases
        self.all_cases_mp = self.product_cases_mp
        self.product_case_pairs = list(zip(self.product_cases, self.product_cases_mp))
        self.product_cases_by_key = {}
        self.product_cases_by_phase_id = {}

    def close_shms(self):
        for case in self.product_cases:
            case.close_and_unlink_shared_memory()

        if self.accumulated_products_shm is not None:
            self.accumulated_products_shm.close()
            self.accumulated_products_shm.unlink()
        if self.product_state_shm is not None:
            self.product_state_shm.close()
            self.product_state_shm.unlink()

        if self.precip_3d_init_shm is not None:
            self.precip_3d_init_shm.close()
            self.precip_3d_init_shm.unlink()
    
    def add_case(self) -> tuple[CaseSetUp, CaseSetUpMP]:
        new_case = CaseSetUp()
        new_case_mp = CaseSetUpMP()
        new_case.is_active = True
        new_case_mp.is_active = True
        self.product_cases.append(new_case)
        self.product_cases_mp.append(new_case_mp)
        self.product_case_pairs = list(zip(self.product_cases, self.product_cases_mp))

    def add_oxidant(self, oxidant):
        if oxidant not in self.all_oxidants:
            self.all_oxidants.append(oxidant)
            self.all_elements.append(oxidant.elem_name)
    
    def add_active(self, active):
        if active not in self.all_actives:
            self.all_actives.append(active)
            self.all_elements.append(active.elem_name)
    
    def add_product(self, product):
        if product not in self.all_products:
            self.all_products.append(product)
