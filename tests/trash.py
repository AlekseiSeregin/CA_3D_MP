import numpy as np

ternary_error = np.array([3, -5, 2, -8, 1])
err = 4

# ternary_error < -err will check which elements of ternary_error are less than -4
result = np.where(ternary_error < -err)

print(result)  # Output: (array([3]),)