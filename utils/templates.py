class CaseSetUp:
    def __init__(self):
        self.oxidant = None
        self.active = None
        self.product = None
        self.to_check_with = None
        self.prod_indexes = None
        self.product_ind_not_stab = None
        self.go_around_func_ref = None
        self.fix_init_precip_func_ref = None
        self.precip_3d_init = None
        self.nucleation_probabilities = None
        self.dissolution_probabilities = None
        self.shm_pool = {"product_indexes": None,
                         "product_ind_not_stab": None,
                         "precip_3d_init": None}


class CaseSetUpMP:
    def __init__(self):
        self.active_c3d_shm_mdata = None
        self.active_cells_shm_mdata = None
        self.active_dirs_shm_mdata = None

        self.oxidant_c3d_shm_mdata = None

        self.product_c3d_shm_mdata = None
        self.full_shm_mdata = None
        self.to_check_with = None
        self.prod_indexes_shm_mdata = None

        self.go_around_func_ref = None
        self.precip_3d_init_shm_mdata = None


class CaseRef:
    def __init__(self):
        self.first = CaseSetUp()
        self.first_mp = CaseSetUpMP()
        self.second = CaseSetUp()
        self.second_mp = CaseSetUpMP()
        self.third = CaseSetUp()
        self.third_mp = CaseSetUpMP()
        self.fourth = CaseSetUp()
        self.fourth_mp = CaseSetUpMP()
