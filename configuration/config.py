from numba.core.utils import T
from .config_utils_classes import ElemInput, GeneratedValues


class Config:
    OXIDANTS = [
        {
            "element": "O",
            "diffusion_condition": "O in Ni Krupp",
            "diffusion_condition_gb": "O in Ni Krupp 100",
            "cells_concentration": 5,
            "diffusion_max_per_cell": 20,
        },
        # {
        #     "element": "N",
        #     "diffusion_condition": "N in Ni Krupp",
        #     "cells_concentration": 0.01,
        #     "diffusion_max_per_cell": 3,
        # },
    ]

    ACTIVES = [
        {
            "element": "Cr",
            "diffusion_condition": "Al in Ni Krupp",
            "mass_concentration": 0.07,
            "cells_concentration": 10,
            "conc_precision": "rand",
            "space_fill": "full",
            "diffusion_max_per_cell": 20,
        },
        # {
        #     "element": "Al",
        #     "diffusion_condition": "Al in Ni Krupp",
        #     "mass_concentration": 0.04,
        #     "cells_concentration": 2,
        #     "conc_precision": "rand",
        #     "space_fill": "full",
        #     "diffusion_max_per_cell": 10,
        # },
    ]

    DEFAULT_PRODUCT_PROBABILITIES = {
        # nucleation
        "p0": 0.01,
        "p0_f": 1,
        "p0_A_const": 1,
        "p0_B_const": 1,
        "p1": 0.3,
        "p1_f": 1,
        "p1_A_const": 1,
        "p1_B_const": 1,
        "global_A": 1,
        "global_B": None,
        "global_B_f": -20,
        "max_neigh_numb": None,
        "nucl_adapt_function": 5,
        # dissolution
        "p0_d": 0.8,
        "p0_d_f": 1,
        "p0_d_A_const": 1,
        "p0_d_B_const": 5,
        "p1_d": 0.7,
        "p1_d_f": 1,
        "p1_d_A_const": 1,
        "p1_d_B_const": 10,
        "p6_d": 1e-6,
        "p6_d_f": 0.99,
        "p6_d_A_const": 1,
        "p6_d_B_const": 20,
        "global_d_A": 1,
        "global_d_B": None,
        "global_d_B_f": -0.33,
        "n": 2,
        "bsf": 3,
        "dissol_adapt_function": 3,
    }

    PRODUCTS = [
        {
            "key": "cr2o3",
            "element": "Cr2O3",
            "jm_identifier": "M2O3",
            # thresholds define nucleation cell consumption
            "threshold_outward": 2,
            "threshold_inward": 3,
            # stoich defines product chemistry/formula
            "stoich": {"Cr": 2, "O": 3},
            "outward_element": "Cr",
            "inward_element": "O",
            "priority": 1,
            "probabilities": dict(DEFAULT_PRODUCT_PROBABILITIES)
        },
        {
            "key": "nicr2o4",
            "element": "NiCr2O4",
            "jm_identifier": "SPINEL_AB2O4",
            "threshold_outward": 2,
            "threshold_inward": 4,
            "stoich": {"Ni": 1, "Cr": 2, "O": 4},
            "outward_element": "Cr",
            "inward_element": "O",
            "priority": 2,
            "probabilities": dict(DEFAULT_PRODUCT_PROBABILITIES)
        },
    ]

    MAP_PRODUCTS_TO_ELEMENTS = False

    MATRIX = ElemInput()

    # matrix
    MATRIX.ELEMENT = "Ni"

    TEMPERATURE = 1100  # °C
    N_CELLS_PER_AXIS = 102  # ONLY MULTIPLES OF 3+(neigh_range-1)*2 ARE ALLOWED
    N_ITERATIONS = 1000000  # must be >= n_cells_per_axis
    STRIDE = 100  # n_iterations / stride = n_iterations for outward diffusion
    STRIDE_MULTIPLIER = 50
    SIM_TIME = 7200  # [sek]
    SIZE = 500 * (10 ** -6)  # [m]

    SOL_PROD = 6.25 * 10 ** -31  # 5.621 * 10 ** -10
    PHASE_FRACTION_LIMIT = 0.036
    THRESHOLD_INWARD = 1
    THRESHOLD_OUTWARD = 1
    NEIGH_RANGE = 1   # neighbouring ranges    1, 2, 3, 4, 5,  6,  7,  8,  9,  10
                      #          and           |  |  |  |  |   |   |   |   |   |
                      # corresponding divisors 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
    N_BOOST_STEPS = 1

    PROD_INCR_CONST = 1 * 10 ** -5
    PROD_ERROR = 0.01
    ZETTA_ZERO = 10 * (10 ** -6)  # [m]
    ZETTA_FINAL = 43 * (10 ** -6)  # [m]

    INWARD_DIFFUSION = True
    OUTWARD_DIFFUSION = True

    OUTWARD_DIFFUSION_WORKERS = 6
    INWARD_DIFFUSION_WORKERS = 2

    # Per-side x boundary (left = x<0, right = x>=n). Read once by diffusion module from Config.
    DIFFUSION_BOUNDARY_X_OUTWARD_LEFT = "deletion"   # periodic | reflection | deletion
    DIFFUSION_BOUNDARY_X_OUTWARD_RIGHT = "reflection"
    DIFFUSION_BOUNDARY_X_INWARD_LEFT = "deletion"
    DIFFUSION_BOUNDARY_X_INWARD_RIGHT = "deletion"
    COMPUTE_PRECIPITATION = True
    SAVE_WHOLE = False
    DECOMPOSE_PRECIPITATIONS = True
    FULL_CELLS = False
    SAVE_PATH = 'C:/test_runs_data/'
    SAVE_POST_PROCESSED_INPUT = True
    USE_SIMPLE_NUCLEATION = False # Legacy switch (kept for backward compatibility)
    # Nucleation kernel mode:
    #   legacy_prob_owner   -> legacy probabilistic with owner-phase exclusion
    #   legacy_simple_owner -> legacy simplified with owner-phase exclusion
    #   stoich_prob_owner   -> threshold-based probabilistic nucleation with owner-phase exclusion
    #   stoich_simple_owner -> threshold-based simplified nucleation with owner-phase exclusion
    NUCLEATION_MODE = "legacy_simple_owner"
    # If True, precipitation stages are executed sequentially by product PRIORITY
    # using entries from PRODUCTS list.
    USE_PRODUCT_STAGE_SEQUENCE = True

    # Execution___________________________________________________________________
    NUMBER_OF_PROCESSES = 10  # Total workers to allocate (split between CA and JMatPro)
    # JMatPro worker allocation ratio (0.0-1.0): fraction of NUMBER_OF_PROCESSES allocated to JMatPro
    # Remaining workers go to CA calculations. Default: auto (60% CA, 40% JMatPro)
    # Examples:
    #   JMATPRO_WORKER_RATIO = 0.4  # 40% JMatPro, 60% CA (recommended for balanced workload)
    #   JMATPRO_WORKER_RATIO = 0.5  # 50/50 split
    #   JMATPRO_WORKER_RATIO = None  # Auto: 40% JMatPro, 60% CA
    JMATPRO_WORKER_RATIO = 0.4  # None = auto allocation
    NUMBER_OF_DIVS_PER_PAGE = 1
    DEPTH_PER_DIV = 1
    MAX_TASK_PER_CHILD = 50000
    TERMINATION_COMMAND = 'd+g+m'
    GENERATED_VALUES = GeneratedValues()
    COMMENT = """NO COMMENTS"""
    INITIAL_SCRIPT = "\n"
