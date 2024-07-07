import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ExpFunct:
    def __init__(self, y0, y1, const_a, const_b, x0=0, x1=1):
        """
        dy > 0, const_a = 1 and const_b > 0: _______/
        dy > 0, const_a = -1 and const_b < 0: /------

        dy < 0, const_a = 1 and const_b < 0: \ _______
        dy < 0, const_a = -1 and const_b > 0: ------\
        """
        dy = y1 - y0

        if (dy > 0 and const_a * const_b < 0) or (dy < 0 and const_a * const_b > 0):
            print("Wrong input into ExpFunct!!!")

        if dy != 0:
            const_c = np.log(dy/(const_a * (np.e**(const_b * x1) - np.e**(const_b * x0))))
            const_d = y0 - const_a * np.e ** (const_b * x0 + const_c)
        else:
            const_c = 0
            const_d = 0
            const_b = 0
            const_a = y0

        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.dy = dy
        self.const_a = const_a
        self.const_b = const_b
        self.const_c = const_c
        self.const_d = const_d

    def evaluate_at_point(self, x):
        return self.const_a * np.e**(self.const_b * x + self.const_c) + self.const_d


# to_check = np.random.randint(10, 100, dtype=np.short)



# _________________________________________________ Gamma Primes
c_a = -1
c_b = -100000

conc_o_m = 0.00263
conc_al_m = 0.0527

k_sp = 6.25 * 10**-31
gamma_m = ((conc_o_m ** 3) * (conc_al_m ** 2)) / k_sp

my_func = ExpFunct(0, 0.3, c_a, c_b)

numb_of_points = 1000

conc_o = np.linspace(0, conc_o_m, numb_of_points)
conc_al = np.linspace(0, conc_al_m, numb_of_points)

conc_mesh = np.meshgrid(conc_o, conc_al)

gamma_primes = (((conc_mesh[0] ** 3) * (conc_mesh[1] ** 2) / k_sp) - 1) / (gamma_m - 1)

p_0 = my_func.evaluate_at_point(gamma_primes)


# Create a meshgrid of x and y coordinates based on the array shape
x_coords, y_coords = np.meshgrid(np.arange(p_0.shape[0]), np.arange(p_0.shape[1]))

# Convert the coordinates and data to 1D arrays
x_coords = x_coords.flatten()
y_coords = y_coords.flatten()
z_coords = p_0.flatten()

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
ax.plot_trisurf(x_coords, y_coords, z_coords, cmap="viridis")

# Set labels for the axes
ax.set_xlabel('X Index')
ax.set_ylabel('Y Index')
ax.set_zlabel('Z Value')

# Show the plot
plt.show()