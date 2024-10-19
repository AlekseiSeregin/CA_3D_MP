import numpy as np
import matplotlib.pyplot as plt
import numba
import progressbar
import math

# Parameters
L = 0.5  # Length of the domain in µm
N = 100  # Number of spatial points
ne = 2  # Number of elements (Oxygen and Manganese)

# Diffusivities for the elements
T = 1073  # temperature in kelvin
D_O = 2.44 * 10 ** 5 * np.exp(-11078 / T)  # Diffusivity of oxygen (µm^2/s)
D_Mn = 7.56 * 10 ** 7 * np.exp(-26983 / T)  # Diffusivity of manganese (µm^2/s)
D = np.array([D_O, D_Mn])

# Molar masses (in g/mol)
M_O = 16  # Oxygen
M_Mn = 55  # Manganese
M_MnO = 16 + 55  # MnO

# Solubility product
K_alpha = 7.19  # Example solubility product for the precipitate BmOp

# Concentrations (initial and boundary conditions)
Csurf_O = 9.23e-3  # Surface concentration of oxygen (in ppm)
Ccore_Mn = 1.235e+4  # Core concentration of manganese (in ppm)

# Calculate Bsurf based on solubility product and surface concentration of oxygen
# Bsurf = (K_BmOp / Csurf_O * p) * (1 / m)  # Surface concentration of manganese based on solubility product

# Spatial discretization
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# Time discretization
dt = min((dx ** 2) / D_O, (dx ** 2) / D_Mn) / 10

s = (dt*D_O)/(dx*dx)

print("dt: ", dt)
print("dx: ", dx)
print("S value: ", s)

# Initial concentrations (ppm)
C = np.zeros((N, ne), dtype=float)


C[:, 1] = Ccore_Mn
C[0, 0] = Csurf_O

CONST_A = M_O * M_Mn / (M_MnO ** 2)

P_alpha = np.zeros(N)

# Time-stepping loop
t_max = 60  # Maximum time in seconds
num_time_steps = int(t_max / dt)
save_interval = max(1, int(t_max / dt))  # Save every 60 seconds


@numba.njit(nopython=True, fastmath=True)
def update_concentration(C, D, dt, dx):
    for i in range(ne):
        for j in range(1, N - 1):
            C[j, i] = C[j, i] + dt * D[i] * (C[j + 1, i] - 2 * C[j, i] + C[j - 1, i]) / dx ** 2
    # Boundary conditions
    # Left boundary (j = 0) assuming Neumann boundary condition (no flux)
    C[0, 0] = Csurf_O
    C[-1, 0] = 0
    C[-1, 1] = Ccore_Mn
    C[0, 1] = C[0, 1] + 2 * dt * D[1] * (C[1, 1] - C[0, 1]) / dx ** 2
    return C

@numba.njit(nopython=True, fastmath=True)
def kinda_lambda(x, b, c):
    return (CONST_A * x ** 2) + (b * x) + c

# Function to compute the derivative of f at point x
@numba.njit(nopython=True, fastmath=True)
def derivative(f, x, b, c, dx=1e-6):
    df = f(x + dx, b, c) - f(x - dx, b, c)
    return df / (2 * dx)
# Newton's method to find roots of the function f

@numba.njit(nopython=True, fastmath=True)
def newton(f, x0, b, c, tol=1e-10, maxit=1000):
    x = x0
    fx = f(x, b, c)

    # Check if the initial guess is close to the root
    if abs(fx) < tol:
        return x

    # Perform iterations
    for _ in range(maxit):
        fpx = derivative(f, x, b, c)  # Compute the derivative of f at x
        if abs(fpx) < tol:  # Avoid division by very small values
            print("Derivative too small")
            break

        x = x - fx / fpx  # Newton's method update
        fx = f(x, b, c)  # Recompute f(x) at new x

        if abs(fx) < tol:  # Check if we're close to the root
            break
    return x

@numba.njit(nopython=True, fastmath=True)
def solve_equilibrium(C, M_Mn, M_O, M_MnO, K_alpha, P_alpha, newton_f, kinda_lambda_f):
    for j in range(N):
        F_O = C[j, 0] + (P_alpha[j] * M_O) / M_MnO
        F_Mn = C[j, 1] + (P_alpha[j] * M_Mn) / M_MnO

        k = C[j, 0] * C[j, 1]

        if k > K_alpha:
            # Solve for P_alpha
            b = -(F_O * M_Mn + F_Mn * M_O) / M_MnO
            c = F_O * F_Mn - K_alpha

            # # Newton's method to find roots of the function f
            # def newton(f, x0, tol=1e-10, maxit=1000):
            #     x = x0
            #     fx = f(x)
            #
            #     # Check if the initial guess is close to the root
            #     if abs(fx) < tol:
            #         return x
            #
            #     # Perform iterations
            #     for _ in range(maxit):
            #         fpx = derivative(f, x)  # Compute the derivative of f at x
            #         if abs(fpx) < tol:  # Avoid division by very small values
            #             print("Derivative too small")
            #             break
            #
            #         x = x - fx / fpx  # Newton's method update
            #         fx = f(x)  # Recompute f(x) at new x
            #
            #         if abs(fx) < tol:  # Check if we're close to the root
            #             break
            #
            #     return x

            # # Define the function using lambda
            # func = lambda x: (CONST_A * x ** 2) + (b * x) + c

            # Initial guess for the root
            x0 = 0

            # Find the root using Newton's method
            P_alpha_calc = newton_f(kinda_lambda_f, x0, b, c, tol=1e-10, maxit=1000)

            # Check if P_alpha is positive
            if P_alpha_calc > 0:
                P_alpha[j] = P_alpha_calc
            else:
                P_alpha[j] = 0

            C[j, 0] = F_O - (P_alpha_calc * M_O) / M_MnO
            C[j, 1] = F_Mn - (P_alpha_calc * M_Mn) / M_MnO

    return P_alpha, C



bar = progressbar.ProgressBar(maxval=num_time_steps).start()
for t in range(num_time_steps):
    bar.update(t)
    # Update concentrations
    C = update_concentration(C, D, dt, dx)
    # Solve equilibrium
    P_alpha, C = solve_equilibrium(C, M_Mn, M_O, M_MnO, K_alpha, P_alpha, newton, kinda_lambda)
bar.finish()



# Distance array for plotting
x = np.linspace(0, L, N)

# Convert to decimal fractions (from ppm)
# F_O_decimal = F_O_save[-1, :] # Convert ppm to mass fraction
F_Mn_decimal = (C[:, 1] + (P_alpha * M_Mn / M_MnO))  # Total Mn includes precipitate
print(C)

for item in P_alpha:
    print(item)

# Plot results
fig = plt.figure(figsize=(12, 12))

# Plot for Oxygen (dissolved mass fraction)
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(x, C[:, 0], label='Dissolved Oxygen', color='blue')
# ax1.set_yscale('log')
# ax1.xlabel('Distance from surface (µm)')
# ax1.ylabel('Mass fraction')
# ax1.title('Dissolved Mass Fraction of Oxygen (ppm)')
# ax1.legend()

# Plot for Manganese (total mass fraction)
ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(x, F_Mn_decimal, label='Free Manganese', color='orange')
# ax2.xlabel('Distance from surface (µm)')
# ax2.ylabel('Mass fraction')
# ax2.title('Total Mass Fraction of Manganese (ppm)')
# ax2.legend()
# ax2.tight_layout()
ax2.set_yscale('log')
ax2.set_ylim(1000, 1000000)

ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(x, P_alpha, label='Total MnO', color='k')

plt.show()

