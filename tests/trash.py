import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfc

# Parameters
L = 0.5  # Length of the domain in micrometers
N = 100  # Number of spatial points
ne = 2  # Number of elements (Oxygen and Manganese)

# Diffusivities for the elements
# Example usage
T = 1073  # temperature in kelvin
D_O = 2.44 * 10 ** 5 * np.exp(-11078 / T)  # Diffusivity of oxygen (µm^2/s)
D_Mn = 7.56 * 10 ** 7 * np.exp(-26983 / T)  # Diffusivity of manganese (µm^2/s)
D = np.array([D_O, D_Mn])

# Molar masses (in g/mol)
M_O = 16  # Oxygen
M_Mn = 55  # Manganese
M_alpha = 71  # Example molecular mass of precipitate (Oxygen + Manganese)

# Solubility product
K_alpha = 7.19  # Example solubility product for the precipitate BmOp

# Concentrations (initial and boundary conditions)
Csurf_O = 9.23e-3  # Surface concentration of oxygen (in ppm)
Ccore_Mn = 1.235e4  # Core concentration of manganese (in ppm)

# Spatial discretization
x = np.linspace(0, L, N)
dx = np.diff(x)
print(f"x: {x}")
print(f"dx: {dx}")

# Time discretization
dt = min(dx) * 2 / max(D_O, D_Mn) / 10
print(f"Time step: {dt}")

# Initial concentrations (ppm)
C = np.zeros((N, ne))
C[:, 1] = Ccore_Mn
C[0, 0] = Csurf_O


# Introducing the mass fraction variable F
F = np.copy(C)


# Function to update concentration
def update_concentration(F, D, dt, dx, P_alpha):
    N = len(F)
    F_new = np.zeros_like(F)
    for i in range(ne):
        for j in range(1, N - 1):
            F_new[j, i] = F[j, i] + dt * D[i] * (
                    ((C[j + 1, i] - C[j, i]) / dx[j]) - ((C[j, i] - C[j - 1, i]) / dx[j - 1])
            ) / dx[j - 1] + dx[j]

        # Boundary conditions
        F_new[0, 1] = F[0, 1] + dt * D[1] * ((C[1, 1] - C[0, 1]) / dx[0] ** 2)
        F_new[-1, i] = C[-1, i]
        F[0, 0] = Csurf_O - (P_alpha[0] * M_O) / M_alpha


    return F_new


# Updated function to solve equilibrium concentration based on the new conditions
def solve_equilibrium(C, M_Mn, M_O, M_MnO, K_alpha):
    N = C.shape[0]
    # P_alpha: ndarray[Any, dtype[floating[_64Bit]]] = np.zeros(N)
    P_alpha = np.zeros(N)

    for j in range(N):
        # Constants
        K_alpha = 7.19
        tolerance = 1e-6
        max_iterations = 100

        # Initial guesses
        C_O = 0.00923
        C_Mn = 12350

        def f(C_O, C_Mn):
            return C_O * C_Mn - K_alpha

        def newton_method(C_O, C_Mn):
            for _ in range(max_iterations):
                f_val = f(C_O, C_Mn)

                # Check if solution is found
                if abs(f_val) < tolerance:
                    break

                # Calculate partial derivatives
                df_dC_O = C_Mn
                df_dC_Mn = C_O

                # Update guesses
                C_O = C_O - f_val / df_dC_O
                C_Mn = C_Mn - f_val / df_dC_Mn

            return C_O, C_Mn

        C_O, C_Mn = newton_method(C_O, C_Mn)
        print(f"Converged values: C_O = {C_O}, C_Mn = {C_Mn}")

        k = C_O * C_Mn

        if k > K_alpha:
            # Solve for P_alpha
            a = M_O * M_Mn / M_MnO ** 2
            b = (C[j, 0] * M_O + C[j, 1] * M_Mn) / M_MnO
            c = C[j, 0] * C[j, 1] - K_alpha

            P_alpha[j] = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            print(P_alpha)

        else:
            P_alpha[j] = 0

    return P_alpha


# Simulation loop
num_time_steps = 1

# Initialize arrays to save results
F_O_save = np.zeros((num_time_steps, N))
F_Mn_save = np.zeros((num_time_steps, N))
P_save = np.zeros((num_time_steps, N))

for t in range(num_time_steps):
    # Solve equilibrium
    P_alpha = solve_equilibrium(F, M_Mn, M_O, K_alpha, M_alpha)
    # Update concentrations
    F = update_concentration(F, D, dt, dx, P_alpha)
    # Store updated mass fractions
    F_O_save[t, :] = F[:, 0]
    F_Mn_save[t, :] = F[:, 1]
    P_save[t, :] = P_alpha



# Plotting the results
plt.figure(figsize=(10, 8))

# Extract the last time step data for plotting
F_Mn_last = F_Mn_save[-1, :]
F_O_last = F_O_save[-1, :]

# Plot for Manganese mass fraction
plt.subplot(2, 1, 1)
plt.plot(x, F_Mn_last, label='F_Mn', color='orange')
plt.xlabel('Distance (μm)')
plt.ylabel('F_Mn (ppm)')
plt.title('Mass Fraction of Manganese')
plt.legend()

# Plot for Oxygen mass fraction
plt.subplot(2, 1, 2)
plt.plot(x, F_O_last, label='F_O', color='blue')
plt.xlabel('Distance (μm)')
plt.ylabel('F_O (ppm)')
plt.title('Mass Fraction of Oxygen')
plt.legend()

plt.tight_layout()
plt.show()
