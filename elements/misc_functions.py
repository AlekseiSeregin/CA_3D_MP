import numpy as np


def fix_full_cells(array_3d, full_array_3d, new_precip, oxid_numb):
    current_precip = np.array(array_3d[new_precip[0], new_precip[1], new_precip[2]], dtype=np.ubyte)
    indexes = np.where(current_precip == oxid_numb)[0]
    full_precip = new_precip[:, indexes]
    full_array_3d[full_precip[0], full_precip[1], full_precip[2]] = True

def dummy(*args, **kwargs):
    pass