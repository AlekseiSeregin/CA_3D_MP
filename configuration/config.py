from .config_utils_classes import ElemInput, ElementGroups, ProdInput, ProdGroups, ConfigProbabilities, GeneratedValues


class Config:
    OXIDANTS = ElementGroups()
    OXIDANTS.PRIMARY = ElemInput()
    OXIDANTS.SECONDARY = ElemInput()

    ACTIVES = ElementGroups()
    ACTIVES.PRIMARY = ElemInput()
    ACTIVES.SECONDARY = ElemInput()

    PRODUCTS = ProdGroups()
    PRODUCTS.PRIMARY = ProdInput()
    PRODUCTS.PRIMARY.PHASE_FRACTION_LIMIT = 2
    PRODUCTS.SECONDARY = ProdInput()
    PRODUCTS.SECONDARY.PHASE_FRACTION_LIMIT = 0.07
    PRODUCTS.TERNARY = ProdInput()
    PRODUCTS.QUATERNARY = ProdInput()

    MATRIX = ElemInput()

    # USER INPUT FROM HERE______________________________________________________________________________________________
    # primary oxidants
    OXIDANTS.PRIMARY.ELEMENT = "O"
    OXIDANTS.PRIMARY.DIFFUSION_CONDITION = "O in Ni Krupp"
    OXIDANTS.PRIMARY.CELLS_CONCENTRATION = 0.0012
    # secondary oxidants
    # OXIDANTS.SECONDARY.ELEMENT = "N"
    # OXIDANTS.SECONDARY.DIFFUSION_CONDITION = "N in Ni Krupp"
    # OXIDANTS.SECONDARY.CELLS_CONCENTRATION = 0.01
    # primary actives
    ACTIVES.PRIMARY.ELEMENT = "Al"
    ACTIVES.PRIMARY.DIFFUSION_CONDITION = "Al in Ni Krupp"
    ACTIVES.PRIMARY.MASS_CONCENTRATION = 0.025
    ACTIVES.PRIMARY.CELLS_CONCENTRATION = 0.2
    # secondary actives
    # ACTIVES.SECONDARY.ELEMENT = "Al"
    # ACTIVES.SECONDARY.DIFFUSION_CONDITION = "Al in Ni Krupp"
    # ACTIVES.SECONDARY.MASS_CONCENTRATION = 0.025
    # ACTIVES.SECONDARY.CELLS_CONCENTRATION = 0.077037037
    # matrix
    MATRIX.ELEMENT = "Ni"

    TEMPERATURE = 1100  # °C
    N_CELLS_PER_AXIS = 102  # ONLY MULTIPLES OF 3+(neigh_range-1)*2 ARE ALLOWED
    N_ITERATIONS = 1000000  # must be >= n_cells_per_axis
    STRIDE = 400  # n_iterations / stride = n_iterations for outward diffusion
    STRIDE_MULTIPLIER = 50
    PRECIP_TRANSFORM_DEPTH = 41
    SIM_TIME = 72000  # [sek]
    SIZE = 100 * (10 ** -6)  # [m]

    SOL_PROD = 6.25 * 10 ** -31  # 5.621 * 10 ** -10
    PHASE_FRACTION_LIMIT = 0.07
    THRESHOLD_INWARD = 1
    THRESHOLD_OUTWARD = 1
    NEIGH_RANGE = 1   # neighbouring ranges    1, 2, 3, 4, 5,  6,  7,  8,  9,  10
                      #          and           |  |  |  |  |   |   |   |   |   |
                      # corresponding divisors 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
    N_BOOST_STEPS = 3

    PROD_INCR_CONST = 1 * 10 ** -5
    PROD_ERROR = 1 * 10 ** -3
    ZETTA_ZERO = 1 * (10 ** -6)  # [m]
    ZETTA_FINAL = 50 * (10 ** -6)  # [m]

    INWARD_DIFFUSION = True
    OUTWARD_DIFFUSION = True
    COMPUTE_PRECIPITATION = True
    SAVE_WHOLE = False
    DECOMPOSE_PRECIPITATIONS = True
    FULL_CELLS = False
    SAVE_PATH = 'C:/test_runs_data/'
    SAVE_POST_PROCESSED_INPUT = True

    # Execution___________________________________________________________________
    MULTIPROCESSING = True
    NUMBER_OF_PROCESSES = 22
    NUMBER_OF_DIVS_PER_PAGE = 2
    DEPTH_PER_DIV = 1
    MAX_TASK_PER_CHILD = 100000

    # PROBABILITIES_______________________________________________________________
    PROBABILITIES = ElementGroups()
    PROBABILITIES.PRIMARY = ConfigProbabilities()
    PROBABILITIES.SECONDARY = ConfigProbabilities()
    # PROBABILITIES.TERNARY = ConfigProbabilities()
    # PROBABILITIES.QUATERNARY = ConfigProbabilities()
    # nucleation primary___________________________
    PROBABILITIES.PRIMARY.p0 = 1
    PROBABILITIES.PRIMARY.p0_f = 1
    PROBABILITIES.PRIMARY.p0_A_const = 1
    PROBABILITIES.PRIMARY.p0_B_const = 1
    PROBABILITIES.PRIMARY.p1 = 0.3
    PROBABILITIES.PRIMARY.p1_f = 1
    PROBABILITIES.PRIMARY.p1_A_const = 1
    PROBABILITIES.PRIMARY.p1_B_const = 1
    PROBABILITIES.PRIMARY.global_A = 1
    PROBABILITIES.PRIMARY.global_B = 1.05
    PROBABILITIES.PRIMARY.global_B_f = -20
    PROBABILITIES.PRIMARY.max_neigh_numb = None
    PROBABILITIES.PRIMARY.nucl_adapt_function = 3
    # dissolution primary_________________________
    PROBABILITIES.PRIMARY.p0_d = 1*10**-2
    PROBABILITIES.PRIMARY.p0_d_f = 1
    PROBABILITIES.PRIMARY.p0_d_A_const = 1
    PROBABILITIES.PRIMARY.p0_d_B_const = 1
    PROBABILITIES.PRIMARY.p1_d = 1*10**-3
    PROBABILITIES.PRIMARY.p1_d_f = 1
    PROBABILITIES.PRIMARY.p1_d_A_const = 1
    PROBABILITIES.PRIMARY.p1_d_B_const = 1
    PROBABILITIES.PRIMARY.p6_d = 1e-6
    PROBABILITIES.PRIMARY.p6_d_f = 0.99
    PROBABILITIES.PRIMARY.p6_d_A_const = 1
    PROBABILITIES.PRIMARY.p6_d_B_const = 1
    PROBABILITIES.PRIMARY.global_d_A = 1
    PROBABILITIES.PRIMARY.global_d_B = -1.7
    PROBABILITIES.PRIMARY.global_d_B_f = -0.33
    PROBABILITIES.PRIMARY.n = 2
    PROBABILITIES.PRIMARY.bsf = 10
    PROBABILITIES.PRIMARY.dissol_adapt_function = 5
    # ________________________

    # nucleation SECONDARY
    PROBABILITIES.SECONDARY.p0 = 0.5
    PROBABILITIES.SECONDARY.p0_f = 1
    PROBABILITIES.SECONDARY.p0_A_const = 1
    PROBABILITIES.SECONDARY.p0_B_const = 1
    PROBABILITIES.SECONDARY.p1 = 0.7
    PROBABILITIES.SECONDARY.p1_f = 1
    PROBABILITIES.SECONDARY.p1_A_const = 1
    PROBABILITIES.SECONDARY.p1_B_const = 1
    PROBABILITIES.SECONDARY.global_A = 1
    PROBABILITIES.SECONDARY.global_B = 0.02
    PROBABILITIES.SECONDARY.global_B_f = -20
    PROBABILITIES.SECONDARY.max_neigh_numb = None
    PROBABILITIES.SECONDARY.nucl_adapt_function = 3
    # dissolution SECONDARY
    PROBABILITIES.SECONDARY.p0_d = 1*10**-3
    PROBABILITIES.SECONDARY.p0_d_f = 1
    PROBABILITIES.SECONDARY.p0_d_A_const = 1
    PROBABILITIES.SECONDARY.p0_d_B_const = 1
    PROBABILITIES.SECONDARY.p1_d = 1*10**-4
    PROBABILITIES.SECONDARY.p1_d_f = 1
    PROBABILITIES.SECONDARY.p1_d_A_const = 1
    PROBABILITIES.SECONDARY.p1_d_B_const = 1
    PROBABILITIES.SECONDARY.p6_d = 1 * 10 ** -7
    PROBABILITIES.SECONDARY.p6_d_f = 0.99
    PROBABILITIES.SECONDARY.p6_d_A_const = 1
    PROBABILITIES.SECONDARY.p6_d_B_const = 1
    PROBABILITIES.SECONDARY.global_d_A = 1
    PROBABILITIES.SECONDARY.global_d_B = -1.7
    PROBABILITIES.SECONDARY.global_d_B_f = -0.001
    PROBABILITIES.SECONDARY.n = 2
    PROBABILITIES.SECONDARY.bsf = 10
    PROBABILITIES.SECONDARY.dissol_adapt_function = 5
    # ________________________

    GENERATED_VALUES = GeneratedValues()
    COMMENT = """NO COMMENTS"""
    INITIAL_SCRIPT = ""
