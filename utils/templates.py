class CaseSetUp:
    def __init__(self):
        self.oxidant = None
        self.active = None
        self.product = None
        self.to_check_with = None
        self.prod_indexes = None
        self.go_around_func_ref = None
        self.fix_init_precip_func_ref = None
        self.precip_3d_init = None
        self.nucleation_probabilities = None
        self.dissolution_probabilities = None


class CaseRef:
    def __init__(self):
        self.first = CaseSetUp()
        self.second = CaseSetUp()
        self.third = CaseSetUp()
        self.fourth = CaseSetUp()
