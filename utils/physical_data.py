from math import *
import numpy as np

DENSITY = {"Ni": 8908, "Fe": 7800, "Ti": 4506, "Cr": 7140, "Al": 2700, "N": 1.17, "O": 1, "None": 1}  # kg/m3
MOLAR_MASS = {"Ti": 0.0479, "Ni": 0.0587, "Fe": 0.0558, "Cr": 0.052, "Al": 0.027, "N": 0.014, "O": 0.016, "TiN": 0.0619,
              "None": 1}  # kg/mol


def get_diff_coeff(temperature, cond):
    diff_coeff = {"N in Ni20Cr2Ti Krupp": 4.7 * 10 ** -6 * exp(-125720 / (8.314 * (273 + temperature))),
                  "O in Ni at 1000 Smithells_Ransley": 2.4 * 10**-13,
                  "O in Ni Krupp": 4.9 * 10 ** -6 * exp(-164000 / (8.314 * (273 + temperature))),
                  "N in Ni Krupp": 1.99 * 10 ** -5 * exp(-132640 / (8.314 * (273 + temperature))),
                  "Al in Ni Krupp": 1.85 * 10 ** -4 * exp(-260780 / (8.314 * (273 + temperature))),
                  "Cr in Ni Krupp": 5.2 * 10 ** -4 * exp(-289000 / (8.314 * (273 + temperature))),
                  "Ti in Ni Krupp": 4.1 * 10 ** -4 * exp(-275000 / (8.314 * (273 + temperature))),
                  "N in Ni Katrin PHD": 1.99 * 10 ** -5 * exp(-132640 / (8.314 * (273 + temperature))),
                  "N in Ni Savva at 1020": 5 * 10 ** -11,
                  "Ti in Ni Savva at 1020": 5 * 10 ** -15,
                  "N in alfa-Fe Rozendaal": 6.6 * 10 ** -7 * exp(-77900 / (8.314 * (273 + temperature))),
                  "Test": 1 * (10 ** -13),
                  "Test_slower": 1.4 * (10 ** -9),
                  "Ti in Ni Krupp boost": 2.13 * 10 ** -13,
                  "Test Diffusion in precipitation": 6.18 * 10 ** -20,
                  "N in FeAlMn": 2.1 * 10 ** -8 * exp(-93517 / (8.314 * (273 + temperature))),
                  "Al in FeAlMn": 2.1 * 10 ** -10 * exp(-93517 / (8.314 * (273 + temperature))),
                  "Cr in NiCr": 0.03 * exp(-40800/(8.314 * (273 + temperature))),
                  "N in NiCr at 800°C": 6.7 * 10 ** -11,
                  # scales_______________________________________________
                  "O in Cr2O3 from [O in Cr2O3]": 15.9 * 10 ** -4 * exp((-100800 * 4.184) / (8.314 * (273 + temperature))),
                  "Cr in Cr2O3 from [Cr in Cr2O3]": 0.137 * 10 ** -4 * exp((-61100 * 4.184) / (8.314 * (273 + temperature))),
                  "None": 1 * (10 ** -13)
                  }
    return diff_coeff[cond]


POWERS = np.array([1.1,
1,
0.91304143,
0.9175948,
0.91774248,
0.911070698,
0.919265669,
0.917351838,
0.91013949,
0.915444787,
0.912218988,
0.919907571,
0.918194976,
0.916890893,
0.915875022,
0.91506829,
0.914416995,
0.913883625,
0.913441341,
0.913070522,
0.912756563,
0.912488406,
0.912257557,
0.912057402,
0.911882733,
0.911729398,
0.91159406,
0.911474009,
0.911367026,
0.911271281,
0.911185254,
0.911107672,
0.911037465,
0.910973728,
0.910915689,
0.910862689,
0.91081416,
0.910769614,
0.910728626,
0.910690828,
0.910655896,
0.910623548,
0.910593536,
0.910565639,
0.910539664,
0.910515438,
0.910492807,
0.910471635,
0.910451798,
0.910433187,
0.910415703,
0.910399256,
0.910383767,
0.910369161,
0.910355374,
0.910342345,
0.91033002,
0.910318348,
0.910307285,
0.910296789,
0.910286821,
0.910277347,
0.910268335,
0.910259756,
0.910251581,
0.910243786,
0.910236348,
0.910229245,
0.910222458,
0.910215967,
0.910209757,
0.910203811,
0.910198114,
0.910192652,
0.910187413,
0.910182386,
0.910177557,
0.910172918,
0.910168459,
0.91016417,
0.910160042,
0.910156068,
0.910152241,
0.910148552,
0.910144996,
0.910141566,
0.910138257,
0.910135062,
0.910131977,
0.910128996,
0.910126115,
0.910123329,
0.910120635,
0.910118028,
0.910115505,
0.910113061,
0.910110695,
0.910108402,
0.910106179,
0.910104024,
0.910101935,
0.9101019355,
0.91010193505,
0.91010193505,
0.91010193505,
0.91010193505,
0.91010193505,
0.91010193505,
0.91010193505,
0.910101935,
0.91010193505])