class ConfigProbabilities:
    def __init__(self):
        # nucleation
        # _________________________
        self.p0 = None
        self.p0_f = None
        self.p0_A_const = None
        self.p0_B_const = None
        self.p1 = None
        self.p1_f = None
        self.p1_A_const = None
        self.p1_B_const = None
        self.global_A = None
        self.global_B = None
        self.global_B_f = None
        self.max_neigh_numb = None
        self.nucl_adapt_function = None
        # _________________________

        # dissolution
        # _________________________
        self.p0_d = None
        self.p0_d_f = None
        self.p0_d_A_const = None
        self.p0_d_B_const = None
        self.p1_d = None
        self.p1_d_f = None
        self.p1_d_A_const = None
        self.p1_d_B_const = None
        self.p6_d = None
        self.p6_d_f = None
        self.p6_d_A_const = None
        self.p6_d_B_const = None
        self.global_d_A = None
        self.global_d_B = None
        self.global_d_B_f = None
        self.bsf = None
        self.dissol_adapt_function = None
        # ________________________


class ElementGroups:
    def __init__(self):
        self.PRIMARY = None
        self.SECONDARY = None


class ElemInput:
    def __init__(self):
        self.ELEMENT = "None"
        self.DIFFUSION_CONDITION = "None"
        self.MASS_CONCENTRATION = 0
        self.CELLS_CONCENTRATION = 0

    def __bool__(self):
        return True


class ProdGroups:
    def __init__(self):
        self.PRIMARY = None
        self.SECONDARY = None
        self.TERNARY = None
        self.QUATERNARY = None

    def __bool__(self):
        return True


class ProdInput:
    def __init__(self):
        pass


class GeneratedValues:
    pass


def get_static_vars_dict(cls):
    # Initialize an empty dictionary
    static_vars_dict = {}
    # Iterate over class attributes
    for attr_name, attr_value in cls.__dict__.items():
        # Exclude special methods and variables starting with '__'
        if not attr_name.startswith('__') and not isinstance(attr_value, classmethod):
            # If the attribute is an instance of another class, recursively convert it to a dictionary
            if isinstance(attr_value, type):
                static_vars_dict[attr_name] = get_static_vars_dict(attr_value)
            else:
                static_vars_dict[attr_name] = attr_value
    return static_vars_dict


def update_class_from_dict(cls, data):
    for key, value in data.items():
        if isinstance(value, dict):
            setattr(cls, key, type(key, (), value))
        else:
            setattr(cls, key, value)


def save_script_contents_as_string(path, cls):
    with open(path, 'r') as file:
        script_content = file.read()
    cls.INITIAL_SCRIPT = "Script: " + path + "\n\n" + script_content
