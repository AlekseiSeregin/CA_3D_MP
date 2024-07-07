import numpy as np
import sys
from configuration import Config


class NucleationProbabilities:
    def __init__(self, param, corresponding_product):
        self.oxidation_number = corresponding_product.OXIDATION_NUMBER
        if not param.max_neigh_numb:
            if self.oxidation_number > 1:
                # self.n_neigh_init = 7 * self.oxidation_number - 1
                self.n_neigh_init = 6 * self.oxidation_number
            else:
                self.n_neigh_init = 6
        else:
            self.n_neigh_init = param.max_neigh_numb

        self.nucl_prob = ExpFunct(Config.N_CELLS_PER_AXIS, param.p0, param.p0_f, param.p0_A_const, param.p0_B_const)
        self.p1 = ExpFunct(Config.N_CELLS_PER_AXIS, param.p1, param.p1_f, param.p1_A_const, param.p1_B_const)

        self.const_a_pp = np.full(Config.N_CELLS_PER_AXIS, param.global_A, dtype=float)
        self.b0 = param.global_B
        self.b1 = param.global_B_f
        self.delt_b = self.b1 - self.b0
        self.const_b_pp = np.full(Config.N_CELLS_PER_AXIS, self.b0, dtype=float)

        # self.left_point = self.oxidation_number
        self.left_point = 1

        self.const_c_pp = np.log((1 - self.p1.values_pp) / (self.const_a_pp *
                                                            (np.e ** (self.const_b_pp * self.n_neigh_init) -
                                                             np.e ** (self.const_b_pp * self.left_point))))
        self.const_d_pp = self.p1.values_pp - self.const_a_pp * np.e ** (self.const_b_pp * self.left_point +
                                                                         self.const_c_pp)
        self.adapt_probabilities = None
        if param.nucl_adapt_function == 0:
            self.adapt_probabilities = self.adapt_nucl_prob
        elif param.nucl_adapt_function == 1:
            self.adapt_probabilities = self.adapt_p1
        elif param.nucl_adapt_function == 2:
            self.adapt_probabilities = self.adapt_p1_nucl_prob
        elif param.nucl_adapt_function == 3:
            self.adapt_probabilities = self.dummy_function

        self.get_probabilities = self.get_probabilities_exp

    def update_constants(self):
        self.const_c_pp = np.log((1 - self.p1.values_pp) / (self.const_a_pp *
                                                            (np.e ** (self.const_b_pp * self.n_neigh_init) -
                                                             np.e ** (self.const_b_pp * self.left_point))))
        self.const_d_pp = self.p1.values_pp - self.const_a_pp * np.e ** (self.const_b_pp * self.left_point +
                                                                         self.const_c_pp)

    def get_probabilities_exp(self, numb_of_neighbours, page_ind):
        return self.const_a_pp[page_ind] * np.e ** (self.const_b_pp[page_ind] * numb_of_neighbours +
                                                    self.const_c_pp[page_ind]) + self.const_d_pp[page_ind]

    def adapt_nucl_prob(self, page_ind, gamma_primes):
        self.nucl_prob.update_values_at_pos(page_ind, gamma_primes)
        self.update_constants()

    def adapt_p1(self, page_ind, rel_phase_fraction):
        self.p1.update_values_at_pos(page_ind, rel_phase_fraction)
        self.const_b_pp[page_ind] = self.delt_b * rel_phase_fraction + self.b0
        self.update_constants()

    def adapt_p1_nucl_prob(self, page_ind, rel_phase_fraction, gamma_primes):
        self.p1.update_values_at_pos(page_ind, rel_phase_fraction)
        self.const_b_pp[page_ind] = self.delt_b * rel_phase_fraction + self.b0

        self.nucl_prob.update_values_at_pos(page_ind, gamma_primes)
        self.update_constants()

    def dummy_function(self, page_ind, rel_phase_fraction):
        pass


class DissolutionProbabilities:
    def __init__(self, param, corresponding_product):
        self.dissol_prob = ExpFunct(Config.N_CELLS_PER_AXIS, param.p0_d, param.p0_d_f,
                                    param.p0_d_A_const, param.p0_d_B_const)
        self.p1 = ExpFunct(Config.N_CELLS_PER_AXIS, param.p1_d, param.p1_d_f,
                           param.p1_d_A_const, param.p1_d_B_const)
        self.min_dissol_prob = ExpFunct(Config.N_CELLS_PER_AXIS, param.p6_d, param.p6_d_f,
                                        param.p6_d_A_const, param.p6_d_B_const)
        self.oxidation_number = corresponding_product.OXIDATION_NUMBER
        # self.n_neigh_init = self.oxidation_number * 5
        self.n_neigh_init = 5
        self.p3 = self.oxidation_number * 3
        self.bsf = param.bsf
        self.const_a_pp = np.full(Config.N_CELLS_PER_AXIS, param.global_d_A, dtype=float)
        self.b0 = param.global_d_B
        self.b1 = param.global_d_B_f
        self.delt_b = self.b1 - self.b0
        self.const_b_pp = np.full(Config.N_CELLS_PER_AXIS, self.b0, dtype=float)

        # self.left_point = self.oxidation_number
        self.left_point = 1

        self.const_c_pp = np.log((self.min_dissol_prob.values_pp - self.p1.values_pp) / (self.const_a_pp *
                                                            (np.e ** (self.const_b_pp * self.n_neigh_init) -
                                                             np.e ** (self.const_b_pp * self.left_point))))
        self.const_d_pp = self.p1.values_pp - self.const_a_pp * np.e ** (self.const_b_pp * self.left_point +
                                                                         self.const_c_pp)

        self.adapt_probabilities = None
        if param.dissol_adapt_function == 0:
            self.adapt_probabilities = self.adapt_dissol_prob
        elif param.dissol_adapt_function == 1:
            self.adapt_probabilities = self.adapt_p1
        elif param.dissol_adapt_function == 2:
            self.adapt_probabilities = self.adapt_dissol_prob_min_dissol_prob
        elif param.dissol_adapt_function == 3:
            self.adapt_probabilities = self.adapt_dissol_prob_min_dissol_prob_p1
        elif param.dissol_adapt_function == 4:
            self.adapt_probabilities = self.adapt_dissol_prob_p1
        elif param.dissol_adapt_function == 5:
            self.adapt_probabilities = self.dummy_function

    def update_constants(self):
        self.const_c_pp = np.log((self.min_dissol_prob.values_pp - self.p1.values_pp) / (self.const_a_pp *
                                                            (np.e ** (self.const_b_pp * self.n_neigh_init) -
                                                             np.e ** (self.const_b_pp * self.left_point))))
        self.const_d_pp = self.p1.values_pp - self.const_a_pp * np.e ** (self.const_b_pp * self.left_point +
                                                                         self.const_c_pp)

    def get_probabilities(self, numb_of_neighbours, page_ind):
        return self.const_a_pp[page_ind] * np.e ** (self.const_b_pp[page_ind] * numb_of_neighbours +
                                                    self.const_c_pp[page_ind]) + self.const_d_pp[page_ind]

    def get_probabilities_block(self, numb_of_neighbours, page_ind):
        return (self.const_a_pp[page_ind] * np.e ** (self.const_b_pp[page_ind] * numb_of_neighbours +
                                                     self.const_c_pp[page_ind]) + self.const_d_pp[page_ind]) / self.bsf

    def adapt_dissol_prob(self, page_ind, rel_phase_fraction):
        self.dissol_prob.update_values_at_pos(page_ind, rel_phase_fraction)
        self.update_constants()

    def adapt_p1(self, page_ind, rel_phase_fraction):
        self.p1.update_values_at_pos(page_ind, rel_phase_fraction)
        self.const_b_pp[page_ind] = self.delt_b * rel_phase_fraction + self.b0

        self.update_constants()

    def adapt_dissol_prob_min_dissol_prob(self, page_ind, rel_phase_fraction):
        self.dissol_prob.update_values_at_pos(page_ind, rel_phase_fraction)
        self.min_dissol_prob.update_values_at_pos(page_ind, rel_phase_fraction)
        self.update_constants()

    def adapt_dissol_prob_min_dissol_prob_p1(self, page_ind, rel_phase_fraction):
        self.dissol_prob.update_values_at_pos(page_ind, rel_phase_fraction)
        self.min_dissol_prob.update_values_at_pos(page_ind, rel_phase_fraction)
        self.p1.update_values_at_pos(page_ind, rel_phase_fraction)
        self.const_b_pp[page_ind] = self.delt_b * rel_phase_fraction + self.b0
        self.update_constants()

    def adapt_dissol_prob_p1(self, page_ind, rel_phase_fraction):
        self.dissol_prob.update_values_at_pos(page_ind, rel_phase_fraction)
        self.p1.update_values_at_pos(page_ind, rel_phase_fraction)
        self.const_b_pp[page_ind] = self.delt_b * rel_phase_fraction + self.b0
        self.update_constants()

    def dummy_function(self, page_ind, rel_phase_fraction):
        pass


class ExpFunct:
    def __init__(self, length, y0, y1, const_a, const_b, x0=0, x1=1):
        """
        dy > 0, const_a = 1 and const_b > 0: _______/
        dy > 0, const_a = -1 and const_b < 0: /------

        dy < 0, const_a = 1 and const_b < 0: \ _______
        dy < 0, const_a = -1 and const_b > 0: ------\
        """
        dy = y1 - y0

        if (dy > 0 > const_a * const_b) or (dy < 0 < const_a * const_b):
            print("Wrong input into ExpFunct!!!")
            sys.exit()

        if dy != 0:
            const_c = np.log(dy/(const_a * (np.e**(const_b * x1) - np.e**(const_b * x0))))
            const_d = y0 - const_a * np.e ** (const_b * x0 + const_c)
        else:
            const_c = 0
            const_d = 0
            const_b = 0
            const_a = y0

        self.length = length
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.dy = dy
        self.const_a = const_a
        self.const_b = const_b
        self.const_c = const_c
        self.const_d = const_d

        self.values_pp = np.full(self.length, self.y0, dtype=float)

    def update_values_at_pos(self, positions, x):
        self.values_pp[positions] = self.const_a * np.e**(self.const_b * x + self.const_c) + self.const_d
