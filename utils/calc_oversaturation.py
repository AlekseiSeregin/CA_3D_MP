from scipy import special
import numpy as np


def calculate_right_side(gamma, phi):
    return np.e**(gamma**2) * special.erf(gamma) /\
             ((phi ** 0.5) * np.e**(phi * gamma**2) * special.erfc(gamma * phi**0.5))


def calc_saturation(c_b):
    d_o = 2.8231080610996937 * 10 ** -12
    d_b = 2.2164389765037816 * 10 ** -14

    c_o = 0.0012
    # c_b = 0.08

    nu = 1

    curr_phi = d_o / d_b
    left_side = c_o / (nu * c_b)

    gammas = np.linspace(0, 1, 10000001)

    res_right_side = []

    for curr_gamma in gammas:
        right_side = calculate_right_side(curr_gamma, curr_phi)

        res_right_side.append(right_side)

    difference = np.absolute(np.subtract(res_right_side, left_side))

    minumum = np.min(difference)

    min_pos = np.where(difference == minumum)[0]

    desired_gamma = gammas[min_pos]

    saturation = 1/((np.pi ** 0.5) * desired_gamma * (curr_phi ** 0.5) * np.e**(curr_phi * desired_gamma**2) * special.erfc(desired_gamma * curr_phi**0.5))

    print(c_b, " ", desired_gamma[0], " ", saturation[0])

conz_list = [0.2]
# conz_list = [0.3]
print()

for active_conz in conz_list:
    calc_saturation(active_conz)
