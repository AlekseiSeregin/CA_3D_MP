from .physical_data import *
import sys
import numpy as np
import time
import datetime
from configuration import Config


class Utils:
    def __init__(self):
        self.param = 0
        # self.db = None
        self.n_cells_per_axis = Config.N_CELLS_PER_AXIS
        self.neigh_range = Config.NEIGH_RANGE

        self.ind_decompose = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1],   # 5 flat
             [1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],      # 9  corners
             [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1],  # 13
             [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],  # 19 side corners
             [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=np.byte)

        self.ind_decompose_no_flat = np.array(
            [[1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],
             [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1],
             [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],
             [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=np.byte)

        self.ind_decompose_flat = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.byte)

        self.ind_decompose_flat_z = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 0]], dtype=np.byte)

        self.ind_formation1 = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 0],
             [1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],
             [-1, 1, -1], [-1, 1, 1], [-1, -1, -1],
             [-1, -1, 1], [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],
             [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=np.byte)

        if self.neigh_range > 1:
            # self.ind_formation = self.generate_neigh_indexes()
            # self.ind_formation = self.generate_neigh_indexes_squashed()
            self.ind_formation = self.generate_neigh_indexes_flat()
        else:
            self.ind_formation = self.generate_neigh_indexes_flat()
            # self.ind_formation = np.array(
            #     [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 0],
            #       [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]], dtype=np.byte)

        self.ind_formation_noz = np.array(np.delete(self.ind_formation1, 13, 0), dtype=np.byte)

        self.interface_neigh = {(0, 0, 1): [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
                                (0, 0, -1): [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1]],
                                (0, 1, 0): [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0]],
                                (0, -1, 0): [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0]],
                                (1, 0, 0): [[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 0, 0]],
                                (-1, 0, 0): [[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [-1, 0, 0]]}


    def generate_param(self):
        Config.GENERATED_VALUES.TAU = Config.SIM_TIME / Config.N_ITERATIONS
        Config.GENERATED_VALUES.LAMBDA = Config.SIZE / Config.N_CELLS_PER_AXIS

        Config.GENERATED_VALUES.KINETIC_KONST = (Config.ZETTA_FINAL-Config.ZETTA_ZERO) / (Config.SIM_TIME ** 0.5)

        if Config.ACTIVES.SECONDARY.ELEMENT == "None":
            Config.ACTIVES.SECONDARY.MASS_CONCENTRATION = 0
            Config.ACTIVES.SECONDARY.CELLS_CONCENTRATION = 0
            Config.ACTIVES.SECONDARY_EXISTENCE = False
        else:
            Config.ACTIVES.SECONDARY_EXISTENCE = True

        if Config.OXIDANTS.SECONDARY.ELEMENT == "None":
            Config.OXIDANTS.SECONDARY.CELLS_CONCENTRATION = 0
            Config.OXIDANTS.SECONDARY_EXISTENCE = False
        else:
            Config.OXIDANTS.SECONDARY_EXISTENCE = True

        self.fetch_from_physical_data()
        self.calc_atomic_conc()
        self.check_c_min_and_calc_ncells()
        self.calc_active_data()
        self.calc_oxidant_data()
        self.calc_product_data()
        self.calc_initial_conc_and_moles()

        time_stamp = int(time.time())
        # Config.GENERATED_VALUES.DB_ID = str(int(time_stamp + random.randint(1, 1000000)))
        Config.GENERATED_VALUES.DB_ID = str(time_stamp)
        Config.GENERATED_VALUES.DB_PATH = Config.SAVE_PATH + Config.GENERATED_VALUES.DB_ID + '.db'
        Config.GENERATED_VALUES.DATE_OF_CREATION = str(datetime.datetime.fromtimestamp(time_stamp))
        print("DB_PATH: ", Config.GENERATED_VALUES.DB_PATH)

        if Config.SAVE_POST_PROCESSED_INPUT:
            path = Config.SAVE_PATH + str(int(time.time())) + '_config.txt'
            with open(path, 'w') as file:
                self.print_static_params_to_file(Config, file)

    @staticmethod
    def calc_product_data():
        # Primary
        Config.PRODUCTS.PRIMARY.MASS_PER_CELL = Config.OXIDANTS.PRIMARY.MASS_PER_CELL + Config.ACTIVES.PRIMARY.MASS_PER_CELL
        Config.PRODUCTS.PRIMARY.MOLES_PER_CELL = Config.ACTIVES.PRIMARY.MOLES_PER_CELL / 2
        Config.PRODUCTS.PRIMARY.MOLES_PER_CELL_TC = Config.PRODUCTS.PRIMARY.MOLES_PER_CELL * 5
        Config.PRODUCTS.PRIMARY.CONSTITUTION = Config.ACTIVES.PRIMARY.ELEMENT + "+" + Config.OXIDANTS.PRIMARY.ELEMENT
        # Secondary
        Config.PRODUCTS.SECONDARY.MASS_PER_CELL = Config.OXIDANTS.PRIMARY.MASS_PER_CELL + Config.ACTIVES.SECONDARY.MASS_PER_CELL
        Config.PRODUCTS.SECONDARY.MOLES_PER_CELL = Config.ACTIVES.SECONDARY.MOLES_PER_CELL / 2
        Config.PRODUCTS.SECONDARY.MOLES_PER_CELL_TC = Config.PRODUCTS.SECONDARY.MOLES_PER_CELL * 5
        Config.PRODUCTS.SECONDARY.CONSTITUTION = Config.ACTIVES.SECONDARY.ELEMENT + "+" + Config.OXIDANTS.PRIMARY.ELEMENT
        # Ternary
        Config.PRODUCTS.TERNARY.MASS_PER_CELL = Config.OXIDANTS.SECONDARY.MASS_PER_CELL + Config.ACTIVES.PRIMARY.MASS_PER_CELL
        Config.PRODUCTS.TERNARY.MOLES_PER_CELL = Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        Config.PRODUCTS.TERNARY.MOLES_PER_CELL_TC = Config.PRODUCTS.TERNARY.MOLES_PER_CELL * 2
        Config.PRODUCTS.TERNARY.CONSTITUTION = Config.ACTIVES.PRIMARY.ELEMENT + "+" + Config.OXIDANTS.SECONDARY.ELEMENT
        # Quaternary
        Config.PRODUCTS.QUATERNARY.MASS_PER_CELL = Config.OXIDANTS.SECONDARY.MASS_PER_CELL + Config.ACTIVES.SECONDARY.MASS_PER_CELL
        Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL = Config.ACTIVES.SECONDARY.MOLES_PER_CELL
        Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL_TC = Config.PRODUCTS.QUATERNARY.MOLES_PER_CELL * 2
        Config.PRODUCTS.QUATERNARY.CONSTITUTION = Config.ACTIVES.SECONDARY.ELEMENT + "+" + Config.OXIDANTS.SECONDARY.ELEMENT

        t_1 = Config.ACTIVES.PRIMARY.MOLAR_MASS * Config.MATRIX.DENSITY / (Config.ACTIVES.PRIMARY.DENSITY * Config.MATRIX.MOLAR_MASS)
        t_2 = Config.ACTIVES.SECONDARY.MOLAR_MASS * Config.MATRIX.DENSITY / (Config.ACTIVES.SECONDARY.DENSITY * Config.MATRIX.MOLAR_MASS)

        Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER =\
            round(Config.MATRIX.MOLES_PER_CELL / (Config.ACTIVES.PRIMARY.MOLES_PER_CELL * t_1))

        if Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER == 1:
            Config.PRODUCTS.PRIMARY.LIND_FLAT_ARRAY = 6
        else:
            Config.PRODUCTS.PRIMARY.LIND_FLAT_ARRAY = 7

        if Config.ACTIVES.SECONDARY_EXISTENCE and Config.OXIDANTS.SECONDARY_EXISTENCE:
            Config.PRODUCTS.SECONDARY.OXIDATION_NUMBER = \
                round(Config.MATRIX.MOLES_PER_CELL / (Config.ACTIVES.SECONDARY.MOLES_PER_CELL * t_2))

            if Config.PRODUCTS.SECONDARY.OXIDATION_NUMBER == 1:
                Config.PRODUCTS.SECONDARY.LIND_FLAT_ARRAY = 6
            else:
                Config.PRODUCTS.SECONDARY.LIND_FLAT_ARRAY = 7

            Config.PRODUCTS.TERNARY.OXIDATION_NUMBER = \
                round(Config.MATRIX.MOLES_PER_CELL / (Config.ACTIVES.PRIMARY.MOLES_PER_CELL * t_1))

            if Config.PRODUCTS.TERNARY.OXIDATION_NUMBER == 1:
                Config.PRODUCTS.TERNARY.LIND_FLAT_ARRAY = 6
            else:
                Config.PRODUCTS.TERNARY.LIND_FLAT_ARRAY = 7

            Config.PRODUCTS.QUATERNARY.OXIDATION_NUMBER = \
                round(Config.MATRIX.MOLES_PER_CELL / (Config.ACTIVES.SECONDARY.MOLES_PER_CELL * t_2))

            if Config.PRODUCTS.QUATERNARY.OXIDATION_NUMBER == 1:
                Config.PRODUCTS.QUATERNARY.LIND_FLAT_ARRAY = 6
            else:
                Config.PRODUCTS.QUATERNARY.LIND_FLAT_ARRAY = 7

        elif Config.ACTIVES.SECONDARY_EXISTENCE and not Config.OXIDANTS.SECONDARY_EXISTENCE:
            Config.PRODUCTS.SECONDARY.OXIDATION_NUMBER = \
                round(Config.MATRIX.MOLES_PER_CELL / (Config.ACTIVES.SECONDARY.MOLES_PER_CELL * t_2))

            if Config.PRODUCTS.SECONDARY.OXIDATION_NUMBER == 1:
                Config.PRODUCTS.SECONDARY.LIND_FLAT_ARRAY = 6
            else:
                Config.PRODUCTS.SECONDARY.LIND_FLAT_ARRAY = 7

    def calc_oxidant_data(self):
        diff_coeff = Config.OXIDANTS.PRIMARY.DIFFUSION_COEFFICIENT
        probabilities = self.calc_prob(diff_coeff)
        Config.OXIDANTS.PRIMARY.PROBABILITIES = probabilities
        Config.OXIDANTS.PRIMARY.PROBABILITIES_2D = self.calc_p0_2d(diff_coeff * 10**1)

        diff_coeff = Config.OXIDANTS.SECONDARY.DIFFUSION_COEFFICIENT
        probabilities = self.calc_prob(diff_coeff)
        Config.OXIDANTS.SECONDARY.PROBABILITIES = probabilities
        Config.OXIDANTS.SECONDARY.PROBABILITIES_2D = self.calc_p0_2d(diff_coeff)

        Config.OXIDANTS.PRIMARY.N_PER_PAGE = round(Config.OXIDANTS.PRIMARY.CELLS_CONCENTRATION * Config.N_CELLS_PER_AXIS ** 2)
        Config.OXIDANTS.SECONDARY.N_PER_PAGE = round(Config.OXIDANTS.SECONDARY.CELLS_CONCENTRATION * Config.N_CELLS_PER_AXIS ** 2)

        Config.OXIDANTS.PRIMARY.MOLES_PER_CELL = Config.ACTIVES.PRIMARY.MOLES_PER_CELL * 1.5 / Config.THRESHOLD_INWARD
        Config.OXIDANTS.PRIMARY.MASS_PER_CELL = Config.OXIDANTS.PRIMARY.MOLES_PER_CELL * Config.OXIDANTS.PRIMARY.MOLAR_MASS

        Config.OXIDANTS.SECONDARY.MOLES_PER_CELL = Config.ACTIVES.SECONDARY.MOLES_PER_CELL / Config.THRESHOLD_INWARD
        Config.OXIDANTS.SECONDARY.MASS_PER_CELL = Config.OXIDANTS.SECONDARY.MOLES_PER_CELL * Config.OXIDANTS.SECONDARY.MOLAR_MASS

    @staticmethod
    def fetch_from_physical_data():
        Config.ACTIVES.PRIMARY.DENSITY = DENSITY[Config.ACTIVES.PRIMARY.ELEMENT]
        Config.ACTIVES.PRIMARY.MOLAR_MASS = MOLAR_MASS[Config.ACTIVES.PRIMARY.ELEMENT]
        Config.ACTIVES.PRIMARY.DIFFUSION_COEFFICIENT = (get_diff_coeff(Config.TEMPERATURE,
                                                                       Config.ACTIVES.PRIMARY.DIFFUSION_CONDITION))
        Config.ACTIVES.SECONDARY.DENSITY = DENSITY[Config.ACTIVES.SECONDARY.ELEMENT]
        Config.ACTIVES.SECONDARY.MOLAR_MASS = MOLAR_MASS[Config.ACTIVES.SECONDARY.ELEMENT]
        Config.ACTIVES.SECONDARY.DIFFUSION_COEFFICIENT = (get_diff_coeff(Config.TEMPERATURE,
                                                                         Config.ACTIVES.SECONDARY.DIFFUSION_CONDITION))
        Config.OXIDANTS.PRIMARY.DENSITY = DENSITY[Config.OXIDANTS.PRIMARY.ELEMENT]
        Config.OXIDANTS.PRIMARY.MOLAR_MASS = MOLAR_MASS[Config.OXIDANTS.PRIMARY.ELEMENT]
        Config.OXIDANTS.PRIMARY.DIFFUSION_COEFFICIENT = (get_diff_coeff(Config.TEMPERATURE,
                                                                        Config.OXIDANTS.PRIMARY.DIFFUSION_CONDITION))
        Config.OXIDANTS.SECONDARY.DENSITY = DENSITY[Config.OXIDANTS.SECONDARY.ELEMENT]
        Config.OXIDANTS.SECONDARY.MOLAR_MASS = MOLAR_MASS[Config.OXIDANTS.SECONDARY.ELEMENT]
        Config.OXIDANTS.SECONDARY.DIFFUSION_COEFFICIENT = (get_diff_coeff(Config.TEMPERATURE,
                                                                          Config.OXIDANTS.SECONDARY.DIFFUSION_CONDITION))
        Config.MATRIX.DENSITY = DENSITY[Config.MATRIX.ELEMENT]
        Config.MATRIX.MOLAR_MASS = MOLAR_MASS[Config.MATRIX.ELEMENT]
        cell_volume = Config.GENERATED_VALUES.LAMBDA ** 3
        Config.MATRIX.MOLES_PER_CELL = Config.MATRIX.DENSITY * cell_volume / Config.MATRIX.MOLAR_MASS
        Config.MATRIX.MASS_PER_CELL = Config.MATRIX.MOLES_PER_CELL * Config.MATRIX.MOLAR_MASS

    def calc_active_data(self):
        Config.ACTIVES.PRIMARY.PROBABILITIES = self.calc_prob(Config.ACTIVES.PRIMARY.DIFFUSION_COEFFICIENT,
                                                              stridden=True)
        Config.ACTIVES.SECONDARY.PROBABILITIES = self.calc_prob(Config.ACTIVES.SECONDARY.DIFFUSION_COEFFICIENT,
                                                                stridden=True)
        matrix_moles = Config.MATRIX.MOLES_PER_CELL
        matrix_molar_mass = Config.MATRIX.MOLAR_MASS
        atomic_c_1 = Config.ACTIVES.PRIMARY.ATOMIC_CONCENTRATION
        atomic_c_2 = Config.ACTIVES.SECONDARY.ATOMIC_CONCENTRATION
        cells_conc1 = Config.ACTIVES.PRIMARY.CELLS_CONCENTRATION
        cells_conc2 = Config.ACTIVES.SECONDARY.CELLS_CONCENTRATION
        molar_mass1 = Config.ACTIVES.PRIMARY.MOLAR_MASS
        molar_mass2 = Config.ACTIVES.SECONDARY.MOLAR_MASS

        t_1 = Config.ACTIVES.PRIMARY.MOLAR_MASS * Config.MATRIX.DENSITY / (Config.ACTIVES.PRIMARY.DENSITY * Config.MATRIX.MOLAR_MASS)
        Config.ACTIVES.PRIMARY.T = t_1
        Config.ACTIVES.PRIMARY.n_ELEM = 1 - t_1
        t_2 = Config.ACTIVES.SECONDARY.MOLAR_MASS * Config.MATRIX.DENSITY / (Config.ACTIVES.SECONDARY.DENSITY * Config.MATRIX.MOLAR_MASS)
        Config.ACTIVES.SECONDARY.T = t_2
        Config.ACTIVES.SECONDARY.n_ELEM = 1 - t_2
        denom = 1 + atomic_c_1 * (t_1 - 1) + atomic_c_2 * (t_2 - 1)

        moles_per_cell1 = atomic_c_1 * matrix_moles / (cells_conc1 * denom)
        Config.ACTIVES.PRIMARY.MOLES_PER_CELL = moles_per_cell1
        Config.ACTIVES.PRIMARY.MASS_PER_CELL = moles_per_cell1 * molar_mass1
        Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL = moles_per_cell1 * t_1
        Config.ACTIVES.PRIMARY.EQ_MATRIX_MASS_PER_CELL = Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL * matrix_molar_mass

        if cells_conc2 > 0:
            moles_per_cell2 = atomic_c_2 * matrix_moles / (cells_conc2 * denom)
        else:
            moles_per_cell2 = 0

        Config.ACTIVES.SECONDARY.MOLES_PER_CELL = moles_per_cell2
        Config.ACTIVES.SECONDARY.MASS_PER_CELL = moles_per_cell2 * molar_mass2
        Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL = moles_per_cell2 * t_2
        Config.ACTIVES.SECONDARY.EQ_MATRIX_MASS_PER_CELL = Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL * matrix_molar_mass

    @staticmethod
    def check_c_min_and_calc_ncells():
        t_1 = Config.ACTIVES.PRIMARY.MOLAR_MASS * Config.MATRIX.DENSITY / (Config.ACTIVES.PRIMARY.DENSITY * Config.MATRIX.MOLAR_MASS)
        t_2 = Config.ACTIVES.SECONDARY.MOLAR_MASS * Config.MATRIX.DENSITY / (Config.ACTIVES.SECONDARY.DENSITY * Config.MATRIX.MOLAR_MASS)
        atomic_c_1 = Config.ACTIVES.PRIMARY.ATOMIC_CONCENTRATION
        atomic_c_2 = Config.ACTIVES.SECONDARY.ATOMIC_CONCENTRATION
        denom = 1 + atomic_c_1 * (t_1 - 1) + atomic_c_2 * (t_2 - 1)

        min_cells1 = atomic_c_1 * t_1 / denom
        if Config.FULL_CELLS:
            Config.ACTIVES.PRIMARY.CELLS_CONCENTRATION = min_cells1
            Config.ACTIVES.PRIMARY.N_PER_PAGE = round(min_cells1 * Config.N_CELLS_PER_AXIS**2)
        elif Config.ACTIVES.PRIMARY.CELLS_CONCENTRATION < min_cells1:
            print()
            print("______________________________________________________________")
            print("Cells Concentration For primary active_element Must Be >= ", min_cells1, "!!!")
            print("______________________________________________________________")
            sys.exit()
        else:
            Config.ACTIVES.PRIMARY.N_PER_PAGE =(
                round(Config.ACTIVES.PRIMARY.CELLS_CONCENTRATION * Config.N_CELLS_PER_AXIS ** 2))

        min_cells2 = atomic_c_2 * t_2 / denom
        if Config.FULL_CELLS:
            if min_cells2 == 0:
                Config.ACTIVES.SECONDARY.N_PER_PAGE = 0
            else:
                Config.ACTIVES.SECONDARY.CELLS_CONCENTRATION = min_cells2
                Config.ACTIVES.SECONDARY.N_PER_PAGE = round(min_cells2 * Config.N_CELLS_PER_AXIS ** 2)
        elif Config.ACTIVES.SECONDARY.CELLS_CONCENTRATION < min_cells2:
            print()
            print("______________________________________________________________")
            print("Cells Concentration For secondary active_element Must Be >= ", min_cells2, "!!!")
            print("______________________________________________________________")
            sys.exit()
        else:
            Config.ACTIVES.SECONDARY.N_PER_PAGE =(
                round(Config.ACTIVES.SECONDARY.CELLS_CONCENTRATION * Config.N_CELLS_PER_AXIS ** 2))

    @staticmethod
    def calc_atomic_conc():
        molar_mass1 = Config.ACTIVES.PRIMARY.MOLAR_MASS
        molar_mass2 = Config.ACTIVES.SECONDARY.MOLAR_MASS
        molar_mass_matr = Config.MATRIX.MOLAR_MASS
        mass_conc1 = Config.ACTIVES.PRIMARY.MASS_CONCENTRATION
        mass_conc2 = Config.ACTIVES.SECONDARY.MASS_CONCENTRATION

        Config.ACTIVES.PRIMARY.ATOMIC_CONCENTRATION = \
            mass_conc1 / (mass_conc1 + (molar_mass1 / molar_mass2) * mass_conc2 + (molar_mass1 / molar_mass_matr) *
                          (1 - mass_conc1 - mass_conc2))

        Config.ACTIVES.SECONDARY.ATOMIC_CONCENTRATION = \
            mass_conc2 / (mass_conc2 + (molar_mass2 / molar_mass1) * mass_conc1 + (molar_mass2 / molar_mass_matr) *
                          (1 - mass_conc1 - mass_conc2))

    @staticmethod
    def calc_initial_conc_and_moles():
        inward = Config.OXIDANTS.PRIMARY.N_PER_PAGE
        inward_moles = inward * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        inward_mass = inward * Config.OXIDANTS.PRIMARY.MASS_PER_CELL

        sinward = Config.OXIDANTS.SECONDARY.N_PER_PAGE
        sinward_moles = sinward * Config.OXIDANTS.SECONDARY.MOLES_PER_CELL
        sinward_mass = sinward * Config.OXIDANTS.SECONDARY.MASS_PER_CELL

        outward = Config.ACTIVES.PRIMARY.N_PER_PAGE
        outward_moles = outward * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_mass = outward * Config.ACTIVES.PRIMARY.MASS_PER_CELL
        outward_eq_mat_moles = outward * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        soutward = Config.ACTIVES.SECONDARY.N_PER_PAGE
        soutward_moles = soutward * Config.ACTIVES.SECONDARY.MOLES_PER_CELL
        soutward_mass = soutward * Config.ACTIVES.SECONDARY.MASS_PER_CELL
        soutward_eq_mat_moles = soutward * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        n_cells_page = (Config.N_CELLS_PER_AXIS ** 2)
        matrix_moles = n_cells_page * Config.MATRIX.MOLES_PER_CELL - outward_eq_mat_moles - soutward_eq_mat_moles
        matrix_mass = matrix_moles * Config.MATRIX.MOLAR_MASS

        whole_moles = matrix_moles + inward_moles + sinward_moles + outward_moles + soutward_moles
        whole_mass = matrix_mass + inward_mass + sinward_mass + outward_mass + soutward_mass

        inward_c_moles = inward_moles / whole_moles
        sinward_c_moles = sinward_moles / whole_moles
        outward_c_moles = outward_moles / whole_moles
        soutward_c_moles = soutward_moles / whole_moles
        matrix_c_moles = matrix_moles / whole_moles

        inward_c_mass = inward_mass / whole_mass
        sinward_c_mass = sinward_mass / whole_mass
        outward_c_mass = outward_mass / whole_mass
        soutward_c_mass = soutward_mass / whole_mass
        matrix_c_mass = matrix_mass / whole_mass
        Config.GENERATED_VALUES.inward_moles = inward_moles
        Config.GENERATED_VALUES.inward_mass = inward_mass
        Config.GENERATED_VALUES.sinward_moles = sinward_moles
        Config.GENERATED_VALUES.sinward_mass = sinward_mass
        Config.GENERATED_VALUES.outward_moles = outward_moles
        Config.GENERATED_VALUES.outward_mass = outward_mass
        Config.GENERATED_VALUES.outward_eq_mat_moles = outward_eq_mat_moles
        Config.GENERATED_VALUES.soutward_moles = soutward_moles
        Config.GENERATED_VALUES.soutward_mass = soutward_mass
        Config.GENERATED_VALUES.soutward_eq_mat_moles = soutward_eq_mat_moles
        Config.GENERATED_VALUES.matrix_moles = matrix_moles
        Config.GENERATED_VALUES.matrix_mass = matrix_mass
        Config.GENERATED_VALUES.whole_moles = whole_moles
        Config.GENERATED_VALUES.whole_mass = whole_mass
        Config.GENERATED_VALUES.inward_c_moles = inward_c_moles
        Config.GENERATED_VALUES.inward_c_mass = inward_c_mass
        Config.GENERATED_VALUES.sinward_c_moles = sinward_c_moles
        Config.GENERATED_VALUES.sinward_c_mass = sinward_c_mass
        Config.GENERATED_VALUES.outward_c_moles = outward_c_moles
        Config.GENERATED_VALUES.outward_c_mass = outward_c_mass
        Config.GENERATED_VALUES.soutward_c_moles = soutward_c_moles
        Config.GENERATED_VALUES.soutward_c_mass = soutward_c_mass
        Config.GENERATED_VALUES.matrix_c_moles = matrix_c_moles
        Config.GENERATED_VALUES.matrix_c_mass = matrix_c_mass
        if Config.SOL_PROD != 0:
            Config.GENERATED_VALUES.max_gamma_min_one = (((inward_c_moles ** 3) * (outward_c_moles ** 2))/Config.SOL_PROD) - 1
        else:
            Config.GENERATED_VALUES.max_gamma_min_one = 0

    @staticmethod
    def calc_prob(diff_coeff, stridden=False):
        if not stridden:
            coeff = 6 * (Config.GENERATED_VALUES.TAU * diff_coeff) / (Config.GENERATED_VALUES.LAMBDA ** 2)
        else:
            new_tau = Config.SIM_TIME / (Config.N_ITERATIONS / Config.STRIDE)
            coeff = 6 * (new_tau * diff_coeff) / (Config.GENERATED_VALUES.LAMBDA ** 2)
        c = (1 - coeff) / (1 + coeff)
        if -(c + 1) / 8 > (c - 1) / 8:
            t = -(c + 1) / 16
        else:
            t = (c - 1) / 16
        p = -2 * t
        p3 = (1 / (coeff + 1)) - 2 * p
        p0 = 1 - 4 * p - p3
        return [p, p3, p0]

    @staticmethod
    def calc_p0_2d(diff_coeff):
        coeff = 4 * (Config.GENERATED_VALUES.TAU * diff_coeff) / (Config.GENERATED_VALUES.LAMBDA ** 2)
        coeff_p = coeff / (1 + coeff)
        if coeff_p < 1:
            r_bound = 1
        else:
            r_bound = 1 / coeff_p
        l_bound = 0
        if 1 - (1/(2*coeff_p)) < 2 - (1/coeff_p):
            if 2 - (1/coeff_p) > 0:
                l_bound = 2 - (1/coeff_p)

        elif 1 - (1/(2*coeff_p)) > 2 - (1/coeff_p):
            if 1 - (1/(2*coeff_p)) > 0:
                l_bound = 1 - (1/(2*coeff_p))
        t = l_bound + (r_bound - l_bound) / 2
        # p = coeff_p * (1 - t)
        p0 = t * coeff_p
        # p3 = 1 + coeff_p * (t - 2)
        return p0

    @staticmethod
    def calc_prob_manually(p, diff_coeff):
        coeff = 6 * (Config.GENERATED_VALUES.TAU * diff_coeff) / (Config.GENERATED_VALUES.LAMBDA ** 2)
        p3 = (1 - 2 * p * (1 + coeff)) / (1 + coeff)
        p0 = 1 - 4 * p - p3
        return {"p": p, "p3": p3, "p0": p0}

    def calc_sur_ind_decompose(self, seeds):
        """
        Calculating the descarts surrounding coordinates for each seed including the position of the seed itself.
        :param seeds: seeds in descarts coordinates
        :return: around_seeds: array of the surrounding coordinates for each seed (26 flat coordinates for each seed)
        """
        seeds = seeds.transpose()
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[item + self.ind_decompose] for item in seeds], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        around_seeds[around_seeds == self.n_cells_per_axis] = 0
        around_seeds[around_seeds == -1] = self.n_cells_per_axis - 1
        return around_seeds

    def calc_sur_ind_decompose_flat(self, seeds):
        """
        Calculating the descarts surrounding coordinates for each seed excluding the position of the seed itself.
        :param seeds: seeds in descarts coordinates
        :return: around_seeds: array of the surrounding coordinates for each seed (6 flat coordinates for each seed)
        """
        seeds = seeds.transpose()
        # generating a neighbouring coordinates for each seed (excluding the position of the seed itself)
        around_seeds = np.array([[item + self.ind_decompose_flat] for item in seeds], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        around_seeds[around_seeds == self.n_cells_per_axis] = 0
        around_seeds[around_seeds == -1] = self.n_cells_per_axis - 1
        return around_seeds

    def calc_sur_ind_decompose_flat_with_zero(self, seeds):
        """
        Calculating the descarts surrounding coordinates for each seed including the position of the seed itself.
        :param seeds: seeds in descarts coordinates
        :return: around_seeds: array of the surrounding coordinates for each seed (7 coordinates for each seed)
        """
        seeds = seeds.transpose()
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[item + self.ind_decompose_flat_z] for item in seeds], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        around_seeds[around_seeds == self.n_cells_per_axis] = 0
        around_seeds[around_seeds == -1] = self.n_cells_per_axis - 1
        return around_seeds

    def calc_sur_ind_decompose_no_flat(self, seeds):
        """
        Calculating the descarts surrounding coordinates for each seed including the position of the seed itself.
        :param seeds: seeds in descarts coordinates
        :return: around_seeds: array of the surrounding coordinates for each seed (7 coordinates for each seed)
        """
        seeds = seeds.transpose()
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[item + self.ind_decompose_no_flat] for item in seeds], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        around_seeds[around_seeds == self.n_cells_per_axis] = 0
        around_seeds[around_seeds == -1] = self.n_cells_per_axis - 1
        return around_seeds

    def calc_sur_ind_formation(self, seeds, dummy_ind):
        """
        Calculating the descarts surrounding coordinates for each seed including the position of the seed itself.
        :param dummy_ind:
        :param seeds: seeds in descarts coordinates
        :return: around_seeds: array of the surrounding coordinates for each seed (26 flat coordinates for each seed)
        """
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[item + self.ind_formation] for item in seeds], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        if seeds[0, 2] < self.neigh_range:
            indexes = np.where(around_seeds[:, :, 2] < 0)
            around_seeds[indexes[0], indexes[1], 2] = dummy_ind
        for shift in range(self.neigh_range):
            indexes = np.where(around_seeds[:, :, 0:2] == self.n_cells_per_axis + shift)
            around_seeds[indexes[0], indexes[1], indexes[2]] = shift
            indexes = np.where(around_seeds[:, :, 0:2] == - shift - 1)
            around_seeds[indexes[0], indexes[1], indexes[2]] = self.n_cells_per_axis - shift - 1
        return around_seeds

    def calc_sur_ind_formation_FLAT(self, seeds, dummy_ind):
        """
        """
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[item + self.ind_formation] for item in seeds], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        if seeds[0, 2] < self.neigh_range:
            indexes = np.where(around_seeds[:, :, 2] < 0)
            around_seeds[indexes[0], indexes[1], 2] = dummy_ind
        for shift in range(self.neigh_range):
            indexes = np.where(around_seeds[:, :, 0:2] == self.n_cells_per_axis + shift)
            around_seeds[indexes[0], indexes[1], indexes[2]] = shift
            indexes = np.where(around_seeds[:, :, 0:2] == - shift - 1)
            around_seeds[indexes[0], indexes[1], indexes[2]] = self.n_cells_per_axis - shift - 1
        return around_seeds

    def calc_sur_ind_formation_noz(self, seeds, dummy_ind):
        """
        Calculating the descarts surrounding coordinates for each seed including the position of the seed itself.
        :param dummy_ind:
        :param seeds: seeds in descarts coordinates
        :return: around_seeds: array of the surrounding coordinates for each seed (26 flat coordinates for each seed)
        """
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[item + self.ind_formation_noz] for item in seeds], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        if seeds[0, 2] < self.neigh_range:
            indexes = np.where(around_seeds[:, :, 2] < 0)
            around_seeds[indexes[0], indexes[1], 2] = dummy_ind
        for shift in range(self.neigh_range):
            indexes = np.where(around_seeds[:, :, 0:2] == self.n_cells_per_axis + shift)
            around_seeds[indexes[0], indexes[1], indexes[2]] = shift
            indexes = np.where(around_seeds[:, :, 0:2] == - shift - 1)
            around_seeds[indexes[0], indexes[1], indexes[2]] = self.n_cells_per_axis - shift - 1
        return around_seeds

    def calc_sur_ind_interface(self, cells, dirs, dummy_ind):
        """
        Calculating the descarts surrounding coordinates for each seed including the position of the seed itself.
        :param dirs:
        :param cells:
        :param dummy_ind:
        :return: around_seeds: array of the surrounding coordinates for each seed (4 flat coordinates for each seed)
        """
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[cell + self.interface_neigh[tuple(dir)]]
                                 for cell, dir in zip(cells.transpose(), dirs.transpose())], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        indexes = np.where(around_seeds[:, :, 2] < 0)
        around_seeds[indexes[0], indexes[1], 2] = dummy_ind

        indexes = np.where(around_seeds[:, :, 0:2] == self.n_cells_per_axis)
        around_seeds[indexes[0], indexes[1], indexes[2]] = 0
        indexes = np.where(around_seeds[:, :, 0:2] == - 1)
        around_seeds[indexes[0], indexes[1], indexes[2]] = self.n_cells_per_axis - 1
        return around_seeds

    def generate_neigh_indexes(self):
        neigh_range = self.neigh_range
        size = 3 + (neigh_range - 1) * 2
        neigh_shape = (size, size, size)
        temp = np.ones(neigh_shape, dtype=int)
        coord = np.array(np.nonzero(temp))
        coord -= neigh_range
        coord = coord.transpose()
        return np.array(coord, dtype=np.byte)

    def generate_neigh_indexes_squashed(self):
        neigh_range = self.neigh_range
        size = 3 + (neigh_range - 1) * 2
        neigh_shape = (size, size, 3)
        temp = np.ones(neigh_shape, dtype=int)

        flat_ind = np.array(self.ind_decompose_flat_z)
        flat_ind = flat_ind.transpose()
        flat_ind[0] += neigh_range
        flat_ind[1] += neigh_range
        flat_ind[2] += 1

        temp[flat_ind[0], flat_ind[1], flat_ind[2]] = 0

        coord = np.array(np.nonzero(temp))
        coord[0] -= neigh_range
        coord[1] -= neigh_range
        coord[2] -= 1
        coord = coord.transpose()

        coord = np.concatenate((self.ind_decompose_flat_z, coord))

        return np.array(coord, dtype=np.byte)

    def generate_neigh_indexes_flat(self):
        neigh_range = self.neigh_range
        size = 3 + (neigh_range - 1) * 2
        neigh_shape = (size, size, 3)
        temp = np.ones(neigh_shape, dtype=int)
        temp[:, :, 0] = 0
        temp[:, :, 2] = 0

        flat_ind = np.array(self.ind_decompose_flat_z)
        flat_ind = flat_ind.transpose()
        flat_ind[0] += neigh_range
        flat_ind[1] += neigh_range
        flat_ind[2] += 1

        temp[flat_ind[0], flat_ind[1], flat_ind[2]] = 0

        coord = np.array(np.nonzero(temp))
        coord[0] -= neigh_range
        coord[1] -= neigh_range
        coord[2] -= 1
        coord = coord.transpose()

        coord = np.concatenate((self.ind_decompose_flat_z, coord))
        additional = coord[[2, 5]]
        coord = np.delete(coord, [2, 5], axis=0)
        coord = np.concatenate((coord, additional))

        return np.array(coord, dtype=np.byte)

    def print_static_params_to_file(self, cls, file_obj, indent=0):
        for attr_name, attr_value in cls.__dict__.items():
            if not callable(attr_value) and not attr_name.startswith('__'):
                if hasattr(attr_value, '__dict__'):
                    file_obj.write(f"{' ' * indent}{attr_name}:\n")
                    self.print_static_params_to_file(attr_value, file_obj, indent + 4)
                else:
                    file_obj.write(f"{' ' * indent}{attr_name}: {attr_value}\n")

    def print_static_params(self, cls, indent=0):
        for attr_name, attr_value in cls.__dict__.items():
            if not callable(attr_value) and not attr_name.startswith('__'):
                if hasattr(attr_value, '__dict__'):
                    print(f"{' ' * indent}{attr_name}:")
                    self.print_static_params(attr_value, indent + 4)
                else:
                    print(f"{' ' * indent}{attr_name}: {attr_value}")
