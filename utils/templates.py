class CaseSetUp:
    def __init__(self):
        self.oxidant = None
        self.active = None
        self.product = None
        self.to_check_with = None
        self.prod_indexes = None
        self.product_ind_not_stab = None
        self.fix_init_precip_func_ref = None
        self.precip_3d_init = None
        self.shm_pool = {"product_indexes": None,
                         "product_ind_not_stab": None,
                         "precip_3d_init": None}

    def close_and_unlink_shared_memory(self):
        for key, shm in self.shm_pool.items():
            if shm is not None:
                shm.close()
                shm.unlink()


class CaseSetUpMP:
    def __init__(self):
        self.active_c3d_shm_mdata = None
        self.active_cells_shm_mdata = None
        self.active_dirs_shm_mdata = None

        self.oxidant_c3d_shm_mdata = None

        self.product_c3d_shm_mdata = None
        self.oxidation_number = None
        self.full_shm_mdata = None
        self.to_check_with_shm_mdata = None
        self.prod_indexes_shm_mdata = None
        self.prod_indexes_not_stab_shm_mdata = None

        self.go_around_func_ref = None
        self.precip_3d_init_shm_mdata = None

        self.nucleation_probabilities = None
        self.dissolution_probabilities = None

        self.precip_step = None
        self.check_intersection = None

        self.decomposition = None
        self.fix_full_cells = None


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

    def close_shms(self):
        self.first.close_and_unlink_shared_memory()
        self.second.close_and_unlink_shared_memory()
        self.third.close_and_unlink_shared_memory()
        self.fourth.close_and_unlink_shared_memory()
