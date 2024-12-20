from scipy import special
import numpy as np


def calculate_right_site(gamma, phi):
    return np.e**(gamma**2) * special.erf(gamma) /\
             ((phi ** 0.5) * np.e**(phi * gamma**2) * special.erfc(gamma * phi**0.5))


def calc_saturation(c_b, c_o, d_o, d_b, time, nu=1.0):
    curr_phi = d_o / d_b
    left_side = c_o / (nu * c_b)
    gammas = np.linspace(0, 1, 100001)
    res_right_side = []
    for curr_gamma in gammas:
        right_side = calculate_right_site(curr_gamma, curr_phi)
        res_right_side.append(right_side)
    difference = np.absolute(np.subtract(res_right_side, left_side))

    nan_ind = np.where(difference > 0)[0]
    difference = difference[nan_ind]

    minumum = np.min(difference)
    min_pos = np.where(difference == minumum)[0]
    desired_gamma = gammas[min_pos]

    saturation = 1/((np.pi ** 0.5) * desired_gamma * (curr_phi ** 0.5) * np.e**(curr_phi * desired_gamma**2) *
                    special.erfc(desired_gamma * curr_phi**0.5))

    depth = 2 * desired_gamma[0] * ((d_o * time)**0.5)
    print(c_b, desired_gamma[0], saturation[0], depth, sep=" ")

# conz_list = [0.25, 0.55, 0.6, 0.65, 0.75, 0.8, 0.85]

conz_list = [0.0561]
c_o = 45e-5
d_o = (2.8231080610996937 * 10 ** -12) * 1
# d_b = 2.2164389765037816 * 10 ** -14
time = 20 * 3600
d_b = (5.258262368514388 * 10 ** -15) * 1
# c_b = 0.08
nu = 1.5
for active_conz in conz_list:
    calc_saturation(active_conz, c_o, d_o, d_b, time, nu=nu)
