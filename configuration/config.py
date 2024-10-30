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
    PRODUCTS.PRIMARY.THRESHOLD_INWARD = 1
    PRODUCTS.PRIMARY.THRESHOLD_OUTWARD = 1

    PRODUCTS.SECONDARY = ProdInput()
    # PRODUCTS.SECONDARY.THRESHOLD_INWARD = 3
    # PRODUCTS.SECONDARY.THRESHOLD_OUTWARD = 2

    PRODUCTS.TERNARY = ProdInput()
    # PRODUCTS.TERNARY.THRESHOLD_INWARD = 4
    # PRODUCTS.TERNARY.THRESHOLD_OUTWARD = 2

    PRODUCTS.QUATERNARY = ProdInput()
    # PRODUCTS.QUATERNARY.THRESHOLD_INWARD = 4
    # PRODUCTS.QUATERNARY.THRESHOLD_OUTWARD = 2

    PRODUCTS.QUINT = ProdInput()
    # PRODUCTS.QUINT.THRESHOLD_INWARD = 1
    # PRODUCTS.QUINT.THRESHOLD_OUTWARD = 0

    MAP_PRODUCTS_TO_ELEMENTS = True

    MATRIX = ElemInput()

    # primary oxidants
    OXIDANTS.PRIMARY.ELEMENT = "O"
    OXIDANTS.PRIMARY.DIFFUSION_CONDITION = "O in Ni Krupp"
    OXIDANTS.PRIMARY.CELLS_CONCENTRATION = 0.001
    # secondary oxidants
    # OXIDANTS.SECONDARY.ELEMENT = "N"
    # OXIDANTS.SECONDARY.DIFFUSION_CONDITION = "N in Ni Krupp"
    # OXIDANTS.SECONDARY.CELLS_CONCENTRATION = 0.01
    # primary actives
    ACTIVES.PRIMARY.ELEMENT = "Al"
    ACTIVES.PRIMARY.DIFFUSION_CONDITION = "Al in Ni Krupp"
    ACTIVES.PRIMARY.MASS_CONCENTRATION = 0.07
    ACTIVES.PRIMARY.CELLS_CONCENTRATION = 0.2
    ACTIVES.PRIMARY.CONC_PRECISION = "exact"
    ACTIVES.PRIMARY.SPACE_FILL = "full"
    # secondary actives
    # ACTIVES.SECONDARY.ELEMENT = "Al"
    # ACTIVES.SECONDARY.DIFFUSION_CONDITION = "Al in Ni Krupp"
    # ACTIVES.SECONDARY.MASS_CONCENTRATION = 0.04
    # ACTIVES.SECONDARY.CELLS_CONCENTRATION = 2
    # 0.0308148148148148000000
    # 0.0770370370370371000000
    # 0.0038518518518518500000
    # 0.0154074074074074000000
    # 0.3851851851851850000000
    # 0.361111111
    # 1.20370370370370000000
    # 1.54074074074074000000
    # 0.92444444444444400000
    # 0.51923076923077
    # 0.96296296296296300000
    # 0.77037037037037000000
    # 10.01481481481480000000

    # matrix
    MATRIX.ELEMENT = "Ni"

    TEMPERATURE = 1100  # °C
    N_CELLS_PER_AXIS = 102  # ONLY MULTIPLES OF 3+(neigh_range-1)*2 ARE ALLOWED
    N_ITERATIONS = 1000000  # must be >= n_cells_per_axis
    STRIDE = 100  # n_iterations / stride = n_iterations for outward diffusion
    STRIDE_MULTIPLIER = 50
    PRECIP_TRANSFORM_DEPTH = 20
    SIM_TIME = 72000  # [sek]
    SIZE = 100 * (10 ** -6)  # [m]

    SOL_PROD = 6.25 * 10 ** -31  # 5.621 * 10 ** -10
    PHASE_FRACTION_LIMIT = 1
    THRESHOLD_INWARD = 1
    THRESHOLD_OUTWARD = 1
    NEIGH_RANGE = 1   # neighbouring ranges    1, 2, 3, 4, 5,  6,  7,  8,  9,  10
                      #          and           |  |  |  |  |   |   |   |   |   |
                      # corresponding divisors 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
    N_BOOST_STEPS = 1

    PROD_INCR_CONST = 1 * 10 ** -5
    PROD_ERROR = 0.01
    ZETTA_ZERO = 1 * (10 ** -6)  # [m]
    ZETTA_FINAL = 50 * (10 ** -6)  # [m]

    INWARD_DIFFUSION = True
    OUTWARD_DIFFUSION = True
    COMPUTE_PRECIPITATION = True
    SAVE_WHOLE = False
    DECOMPOSE_PRECIPITATIONS = False
    FULL_CELLS = False
    SAVE_PATH = 'C:/test_runs_data/'
    SAVE_POST_PROCESSED_INPUT = True

    # Execution___________________________________________________________________
    MULTIPROCESSING = True
    NUMBER_OF_PROCESSES = 24
    NUMBER_OF_DIVS_PER_PAGE = 1
    DEPTH_PER_DIV = 1
    MAX_TASK_PER_CHILD = 50000
    BUFF_SIZE_CONST_ELEM = 1.5
    TERMINATION_COMMAND = 'ctrl+g+m'

    # PROBABILITIES_______________________________________________________________
    PROBABILITIES = ElementGroups()
    PROBABILITIES.PRIMARY = ConfigProbabilities()
    PROBABILITIES.SECONDARY = ConfigProbabilities()
    # PROBABILITIES.TERNARY = ConfigProbabilities()
    # PROBABILITIES.QUATERNARY = ConfigProbabilities()
    # PROBABILITIES.QUINT = ConfigProbabilities()

    # nucleation primary___________________________
    # PROBABILITIES.PRIMARY.p0 = 0.01
    # PROBABILITIES.PRIMARY.p0_f = 1
    # PROBABILITIES.PRIMARY.p0_A_const = 1
    # PROBABILITIES.PRIMARY.p0_B_const = 1
    # PROBABILITIES.PRIMARY.p1 = 0.3
    # PROBABILITIES.PRIMARY.p1_f = 1
    # PROBABILITIES.PRIMARY.p1_A_const = 1
    # PROBABILITIES.PRIMARY.p1_B_const = 1
    # PROBABILITIES.PRIMARY.global_A = 1
    # PROBABILITIES.PRIMARY.global_B = None
    # PROBABILITIES.PRIMARY.global_B_f = -20
    # PROBABILITIES.PRIMARY.max_neigh_numb = None
    # PROBABILITIES.PRIMARY.nucl_adapt_function = 5
    # # dissolution primary_________________________
    # PROBABILITIES.PRIMARY.p0_d = 0.01
    # PROBABILITIES.PRIMARY.p0_d_f = 1
    # PROBABILITIES.PRIMARY.p0_d_A_const = 1
    # PROBABILITIES.PRIMARY.p0_d_B_const = 5
    # PROBABILITIES.PRIMARY.p1_d = 0.005
    # PROBABILITIES.PRIMARY.p1_d_f = 1
    # PROBABILITIES.PRIMARY.p1_d_A_const = 1
    # PROBABILITIES.PRIMARY.p1_d_B_const = 10
    # PROBABILITIES.PRIMARY.p6_d = 1e-5
    # PROBABILITIES.PRIMARY.p6_d_f = 0.99
    # PROBABILITIES.PRIMARY.p6_d_A_const = 1
    # PROBABILITIES.PRIMARY.p6_d_B_const = 20
    # PROBABILITIES.PRIMARY.global_d_A = 1
    # PROBABILITIES.PRIMARY.global_d_B = None
    # PROBABILITIES.PRIMARY.global_d_B_f = -0.33
    # PROBABILITIES.PRIMARY.n = 2
    # PROBABILITIES.PRIMARY.bsf = 1
    # PROBABILITIES.PRIMARY.dissol_adapt_function = 3
    # ________________________

    # nucleation SECONDARY
    # PROBABILITIES.SECONDARY.p0 = 0.001
    # PROBABILITIES.SECONDARY.p0_f = 1
    # PROBABILITIES.SECONDARY.p0_A_const = 1
    # PROBABILITIES.SECONDARY.p0_B_const = 1
    # PROBABILITIES.SECONDARY.p1 = 0.01
    # PROBABILITIES.SECONDARY.p1_f = 1
    # PROBABILITIES.SECONDARY.p1_A_const = 1
    # PROBABILITIES.SECONDARY.p1_B_const = 1
    # PROBABILITIES.SECONDARY.global_A = 1
    # PROBABILITIES.SECONDARY.global_B = None
    # PROBABILITIES.SECONDARY.global_B_f = -20
    # PROBABILITIES.SECONDARY.max_neigh_numb = None
    # PROBABILITIES.SECONDARY.nucl_adapt_function = 5
    # # dissolution SECONDARY
    # PROBABILITIES.SECONDARY.p0_d = 0.1
    # PROBABILITIES.SECONDARY.p0_d_f = 1
    # PROBABILITIES.SECONDARY.p0_d_A_const = 1
    # PROBABILITIES.SECONDARY.p0_d_B_const = 1
    # PROBABILITIES.SECONDARY.p1_d = 0.01
    # PROBABILITIES.SECONDARY.p1_d_f = 1
    # PROBABILITIES.SECONDARY.p1_d_A_const = 1
    # PROBABILITIES.SECONDARY.p1_d_B_const = 1
    # PROBABILITIES.SECONDARY.p6_d = 1 * 10 ** -4
    # PROBABILITIES.SECONDARY.p6_d_f = 0.99
    # PROBABILITIES.SECONDARY.p6_d_A_const = 1
    # PROBABILITIES.SECONDARY.p6_d_B_const = 1
    # PROBABILITIES.SECONDARY.global_d_A = 1
    # PROBABILITIES.SECONDARY.global_d_B = None
    # PROBABILITIES.SECONDARY.global_d_B_f = -0.001
    # PROBABILITIES.SECONDARY.n = 2
    # PROBABILITIES.SECONDARY.bsf = 10
    # PROBABILITIES.SECONDARY.dissol_adapt_function = 3
    # # ________________________
    #
    # # nucleation TERNARY
    # PROBABILITIES.TERNARY.p0 = 0.1
    # PROBABILITIES.TERNARY.p0_f = 1
    # PROBABILITIES.TERNARY.p0_A_const = 1
    # PROBABILITIES.TERNARY.p0_B_const = 1
    # PROBABILITIES.TERNARY.p1 = 0.3
    # PROBABILITIES.TERNARY.p1_f = 1
    # PROBABILITIES.TERNARY.p1_A_const = 1
    # PROBABILITIES.TERNARY.p1_B_const = 1
    # PROBABILITIES.TERNARY.global_A = 1
    # PROBABILITIES.TERNARY.global_B = None
    # PROBABILITIES.TERNARY.global_B_f = -20
    # PROBABILITIES.TERNARY.max_neigh_numb = None
    # PROBABILITIES.TERNARY.nucl_adapt_function = 3
    # # dissolution TERNARY
    # PROBABILITIES.TERNARY.p0_d = 1 * 10 ** -3
    # PROBABILITIES.TERNARY.p0_d_f = 1
    # PROBABILITIES.TERNARY.p0_d_A_const = 1
    # PROBABILITIES.TERNARY.p0_d_B_const = 1
    # PROBABILITIES.TERNARY.p1_d = 1 * 10 ** -4
    # PROBABILITIES.TERNARY.p1_d_f = 1
    # PROBABILITIES.TERNARY.p1_d_A_const = 1
    # PROBABILITIES.TERNARY.p1_d_B_const = 1
    # PROBABILITIES.TERNARY.p6_d = 1 * 10 ** -7
    # PROBABILITIES.TERNARY.p6_d_f = 0.99
    # PROBABILITIES.TERNARY.p6_d_A_const = 1
    # PROBABILITIES.TERNARY.p6_d_B_const = 1
    # PROBABILITIES.TERNARY.global_d_A = 1
    # PROBABILITIES.TERNARY.global_d_B = None
    # PROBABILITIES.TERNARY.global_d_B_f = -0.001
    # PROBABILITIES.TERNARY.n = 2
    # PROBABILITIES.TERNARY.bsf = 10
    # PROBABILITIES.TERNARY.dissol_adapt_function = 3
    # # ________________________
    #
    # # nucleation QUATERNARY
    # PROBABILITIES.QUATERNARY.p0 = 0.1
    # PROBABILITIES.QUATERNARY.p0_f = 1
    # PROBABILITIES.QUATERNARY.p0_A_const = 1
    # PROBABILITIES.QUATERNARY.p0_B_const = 1
    # PROBABILITIES.QUATERNARY.p1 = 0.3
    # PROBABILITIES.QUATERNARY.p1_f = 1
    # PROBABILITIES.QUATERNARY.p1_A_const = 1
    # PROBABILITIES.QUATERNARY.p1_B_const = 1
    # PROBABILITIES.QUATERNARY.global_A = 1
    # PROBABILITIES.QUATERNARY.global_B = None
    # PROBABILITIES.QUATERNARY.global_B_f = -20
    # PROBABILITIES.QUATERNARY.max_neigh_numb = None
    # PROBABILITIES.QUATERNARY.nucl_adapt_function = 3
    # # dissolution QUATERNARY
    # PROBABILITIES.QUATERNARY.p0_d = 1 * 10 ** -3
    # PROBABILITIES.QUATERNARY.p0_d_f = 1
    # PROBABILITIES.QUATERNARY.p0_d_A_const = 1
    # PROBABILITIES.QUATERNARY.p0_d_B_const = 1
    # PROBABILITIES.QUATERNARY.p1_d = 1 * 10 ** -4
    # PROBABILITIES.QUATERNARY.p1_d_f = 1
    # PROBABILITIES.QUATERNARY.p1_d_A_const = 1
    # PROBABILITIES.QUATERNARY.p1_d_B_const = 1
    # PROBABILITIES.QUATERNARY.p6_d = 1 * 10 ** -7
    # PROBABILITIES.QUATERNARY.p6_d_f = 0.99
    # PROBABILITIES.QUATERNARY.p6_d_A_const = 1
    # PROBABILITIES.QUATERNARY.p6_d_B_const = 1
    # PROBABILITIES.QUATERNARY.global_d_A = 1
    # PROBABILITIES.QUATERNARY.global_d_B = None
    # PROBABILITIES.QUATERNARY.global_d_B_f = -0.001
    # PROBABILITIES.QUATERNARY.n = 2
    # PROBABILITIES.QUATERNARY.bsf = 10
    # PROBABILITIES.QUATERNARY.dissol_adapt_function = 3
    # # ________________________
    #
    # # nucleation QUINT
    # PROBABILITIES.QUINT.p0 = 0.1
    # PROBABILITIES.QUINT.p0_f = 1
    # PROBABILITIES.QUINT.p0_A_const = 1
    # PROBABILITIES.QUINT.p0_B_const = 1
    # PROBABILITIES.QUINT.p1 = 0.3
    # PROBABILITIES.QUINT.p1_f = 1
    # PROBABILITIES.QUINT.p1_A_const = 1
    # PROBABILITIES.QUINT.p1_B_const = 1
    # PROBABILITIES.QUINT.global_A = 1
    # PROBABILITIES.QUINT.global_B = None
    # PROBABILITIES.QUINT.global_B_f = -20
    # PROBABILITIES.QUINT.max_neigh_numb = None
    # PROBABILITIES.QUINT.nucl_adapt_function = 3
    # # dissolution QUINT
    # PROBABILITIES.QUINT.p0_d = 1 * 10 ** -3
    # PROBABILITIES.QUINT.p0_d_f = 1
    # PROBABILITIES.QUINT.p0_d_A_const = 1
    # PROBABILITIES.QUINT.p0_d_B_const = 1
    # PROBABILITIES.QUINT.p1_d = 1 * 10 ** -4
    # PROBABILITIES.QUINT.p1_d_f = 1
    # PROBABILITIES.QUINT.p1_d_A_const = 1
    # PROBABILITIES.QUINT.p1_d_B_const = 1
    # PROBABILITIES.QUINT.p6_d = 1 * 10 ** -7
    # PROBABILITIES.QUINT.p6_d_f = 0.99
    # PROBABILITIES.QUINT.p6_d_A_const = 1
    # PROBABILITIES.QUINT.p6_d_B_const = 1
    # PROBABILITIES.QUINT.global_d_A = 1
    # PROBABILITIES.QUINT.global_d_B = None
    # PROBABILITIES.QUINT.global_d_B_f = -0.001
    # PROBABILITIES.QUINT.n = 2
    # PROBABILITIES.QUINT.bsf = 10
    # PROBABILITIES.QUINT.dissol_adapt_function = 3
    # # ________________________

    GENERATED_VALUES = GeneratedValues()
    COMMENT = """NO COMMENTS"""
    INITIAL_SCRIPT = "\n"
