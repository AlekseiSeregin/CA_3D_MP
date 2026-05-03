from .config_utils_classes import ElemInput, GeneratedValues


class Config:
    OXIDANTS = [
        {
            "element": "O",
            "diffusion_condition": "O in Ni Krupp",
            "diffusion_condition_gb": "O in Ni Krupp 100",
            # Optional: diffusion through product cells for this species (condition name from physical_data.py).
            # You can also provide "diffusion_coefficient_in_product": <float>.
            "diffusion_condition_in_product": "O in Cr2O3 from [O in Cr2O3]",
            "cells_concentration": 40,
            "diffusion_max_per_cell": 400,
            "atomic_fraction": 0.11464568414589577
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
            "diffusion_condition": "Cr in Ni Krupp",
            # Optional: species-level fallback used inside product cells unless product-specific override is defined.
            "diffusion_condition_in_product": None,
            "mass_concentration": 0.25,
            "cells_concentration": 80,
            "conc_precision": "rand",
            "space_fill": "full",
            "diffusion_max_per_cell": 400,
        },
        {
            "element": "Al",
            "diffusion_condition": "Al in Ni Krupp",
            "mass_concentration": 0.025,
            "cells_concentration": 15.40740741,
            "conc_precision": "rand",
            "space_fill": "full",
            "diffusion_max_per_cell": 100,
            # 5.192307693
        },
    ]

    DEFAULT_PRODUCT_PROBABILITIES = {
        # nucleation
        "p0": 0.1,
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
        "p0_d": 0.6,
        "p0_d_f": 1,
        "p0_d_A_const": 1,
        "p0_d_B_const": 5,
        "p1_d": 0.4,
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
        "n": 0,
        "bsf": 0,
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
            # Optional per-product overrides for diffusion through this product.
            # Map diffusing element -> either condition string or numeric coefficient.
            # Example: {"O": "O in Cr2O3 from [O in Cr2O3]", "Cr": "Cr in Cr2O3 from [Cr in Cr2O3]"}
            "diffusion_in_product": {},
            "priority": 3,
            "probabilities": dict(DEFAULT_PRODUCT_PROBABILITIES),
            # Defines the number of iteration steps relative to the total number of iterations in which the product will be dissolved assumed that the curent product fraction is in the in the error range.
            # Attempt to implement a ostwald ripening effect and round up the particle form.
            "dissolution_time_ratio": 0.01
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
            "priority": 4,
            "probabilities": dict(DEFAULT_PRODUCT_PROBABILITIES),
            # Defines the number of iteration steps relative to the total number of iterations in which the product will be dissolved assumed that the curent product fraction is in the in the error range.
            # Attempt to implement a ostwald ripening effect and round up the particle form.
            "dissolution_time_ratio": 0.001
        },
        {
            "key": "al2o3",
            "element": "Al2O3",
            "jm_identifier": "M2O3",
            # thresholds define nucleation cell consumption
            "threshold_outward": 2,
            "threshold_inward": 3,
            # stoich defines product chemistry/formula
            "stoich": {"Al": 2, "O": 3},
            "outward_element": "Al",
            "inward_element": "O",
            "priority": 1,
            "probabilities": dict(DEFAULT_PRODUCT_PROBABILITIES)
        },
        {
            "key": "nial2o4",
            "element": "NiAl2O4",
            "jm_identifier": "SPINEL_AB2O4",
            "threshold_outward": 2,
            "threshold_inward": 4,
            "stoich": {"Ni": 1, "Al": 2, "O": 4},
            "outward_element": "Al",
            "inward_element": "O",
            "priority": 2,
            "probabilities": dict(DEFAULT_PRODUCT_PROBABILITIES)
        },
        {
            "key": "nio",
            "element": "NiO",
            "jm_identifier": "MO_B2",
            "threshold_outward": 0,
            "threshold_inward": 1,
            "stoich": {"Ni": 1, "O": 1},
            "outward_element": None,
            "inward_element": "O",
            "priority": 5,
            "probabilities": dict(DEFAULT_PRODUCT_PROBABILITIES),
            # Defines the number of iteration steps relative to the total number of iterations in which the product will be dissolved assumed that the curent product fraction is in the in the error range.
            # Attempt to implement a ostwald ripening effect and round up the particle form.
            "dissolution_time_ratio": 0.001
        },
    ]

    MAP_PRODUCTS_TO_ELEMENTS = False

    MATRIX = ElemInput()

    # matrix
    MATRIX.ELEMENT = "Ni"

    TEMPERATURE = 1100  # °C
    N_CELLS_PER_AXIS = 100  # ONLY MULTIPLES OF 3+(neigh_range-1)*2 ARE ALLOWED
    N_ITERATIONS = 100000  # must be >= n_cells_per_axis
    STRIDE = 100  # n_iterations / stride = n_iterations for outward diffusion
    STRIDE_MULTIPLIER = 50
    SIM_TIME = 72000  # [sek]
    SIZE = 500 * (10 ** -6)  # [m]

    SOL_PROD = 6.25 * 10 ** -31  # 5.621 * 10 ** -10
    PHASE_FRACTION_LIMIT = 0.036
    NEIGH_RANGE = 1   # neighbouring ranges    1, 2, 3, 4, 5,  6,  7,  8,  9,  10
                      #          and           |  |  |  |  |   |   |   |   |   |
                      # corresponding divisors 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
    N_BOOST_STEPS = 1

    PROD_INCR_CONST = 1 * 10 ** -5
    PROD_ERROR = 0.1
    ZETTA_ZERO = 10 * (10 ** -6)  # [m]
    ZETTA_FINAL = 43 * (10 ** -6)  # [m]

    INWARD_DIFFUSION = True
    OUTWARD_DIFFUSION = True

    OUTWARD_DIFFUSION_WORKERS = 10
    INWARD_DIFFUSION_WORKERS = 10

    # Per-side x boundary (left = x<0, right = x>=n). Read once by diffusion module from Config.
    DIFFUSION_BOUNDARY_X_OUTWARD_LEFT = "reflection"   # periodic | reflection | deletion
    DIFFUSION_BOUNDARY_X_OUTWARD_RIGHT = "reflection"
    DIFFUSION_BOUNDARY_X_INWARD_LEFT = "deletion"
    DIFFUSION_BOUNDARY_X_INWARD_RIGHT = "deletion"
    COMPUTE_PRECIPITATION = True
    PRECIPITATION_STRIDE = 1
    SAVE_WHOLE = False
    DECOMPOSE_PRECIPITATIONS = False
    FULL_CELLS = False
    SAVE_PATH = 'C:/test_runs_data/'
    SAVE_POST_PROCESSED_INPUT = True
    RECALC_ELEM_COUNTS_FROM_PRODUCT = True

    # Nucleation kernel mode:
    #   legacy_prob_owner   -> legacy probabilistic with owner-phase exclusion
    #   legacy_simple_owner -> legacy simplified with owner-phase exclusion
    #   stoich_prob_owner   -> threshold-based probabilistic nucleation with owner-phase exclusion
    #   stoich_simple_owner -> threshold-based simplified nucleation with owner-phase exclusion
    NUCLEATION_MODE = "stoich_prob_owner"

    # If True, use fold nucleation kernels: new product is placed on the fullest non-full cell
    # among center + 6 face neighbours of the reaction cell (snapshot neighbour logic unchanged).
    # When flat_count == 0 (no product in neighbour stencil on product_init), placement stays on the oxidant cell.
    # Only supported for legacy_prob_owner and stoich_prob_owner (including no-outward _SPEC variants).
    NUCLEATION_APPLY_FOLD = True

    # Same-plane fold sub-flag. Only meaningful when NUCLEATION_APPLY_FOLD is True.
    # When True, the fold target is restricted to the seed cell's x-plane: only the
    # center and the 4 in-plane face neighbours (±y, ±z) are considered, the ±x
    # neighbours are skipped. This prevents nucleation events on plane i from
    # raising the product count on plane i±1.
    NUCLEATION_FOLD_SAME_PLANE = True

    
    # Execution___________________________________________________________________
    NUMBER_OF_PROCESSES = 10  # Total workers to allocate (split between CA and JMatPro)
    # JMatPro worker allocation ratio (0.0-1.0): fraction of NUMBER_OF_PROCESSES allocated to JMatPro
    # Remaining workers go to CA calculations. Default: auto (60% CA, 40% JMatPro)
    # Examples:
    #   JMATPRO_WORKER_RATIO = 0.4  # 40% JMatPro, 60% CA (recommended for balanced workload)
    #   JMATPRO_WORKER_RATIO = 0.5  # 50/50 split
    #   JMATPRO_WORKER_RATIO = None  # Auto: 40% JMatPro, 60% CA
    JMATPRO_WORKER_RATIO = 1 # None = auto allocation

    # JMatPro composition sampling mode:
    # If True, compute compositions per 3D subblock (blocks_per_axis^3 total) but only for
    # ignited blocks along x up to the furthest inward particle. Non-ignited blocks are left unchanged.
    USE_JMATPRO_BLOCKS_IGNITED = False

    # --- JMatPro block geometry (two modes; code prefers explicit cell sizes when set) ---
    # Preferred: cells per block along each axis. If X, Y, and Z are all > 0, the grid uses
    #   Bx = N_CELLS_PER_AXIS / X,  By = N / Y,  Bz = N / Z
    # and JMATPRO_BLOCKS_PER_AXIS is ignored. Requires N divisible by each of X,Y,Z.
    # Z-bitmask storage requires Bz <= 16 (raise if violated).
    JMATPRO_BLOCK_CELLS_X = 1
    JMATPRO_BLOCK_CELLS_Y = 20
    JMATPRO_BLOCK_CELLS_Z = 20

    MAX_TASK_PER_CHILD = 400
    TERMINATION_COMMAND = 'd+g+m'

    # Per-iteration text logging of JMatPro state. When enabled, every call to the
    # JMatPro lookup writes a per-plane block to a text file (free atoms, bound
    # atoms in products, product cells, composition fed to JMatPro, existing and
    # JMatPro-target product concentrations). Useful for diagnosing why higher-O
    # phases appear after many nucleate/dissolve cycles.
    LOG_PER_ITER_TEXT = False
    LOG_PER_ITER_PATH = ""  # Empty string -> SAVE_PATH/per_iter_jmatpro_log.txt

    GENERATED_VALUES = GeneratedValues()
    INITIAL_SCRIPT = "\n"
