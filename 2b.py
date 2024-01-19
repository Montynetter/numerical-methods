import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Runge-Kutta 4th order method
def runge_kutta_4th_order(f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    h = t[1] - t[0]  # Assuming uniform step size
    for i in range(1, len(t)):
        k1 = h * f(y[i-1], t[i-1])
        k2 = h * f(y[i-1] + 0.5 * k1, t[i-1] + 0.5 * h)
        k3 = h * f(y[i-1] + 0.5 * k2, t[i-1] + 0.5 * h)
        k4 = h * f(y[i-1] + k3, t[i-1] + h)
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y

# Euler-Cauchy method
def euler_cauchy_method(f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    h = t[1] - t[0]  # Assuming uniform step size
    for i in range(1, len(t)):
        y_predict = y[i-1] + h * f(y[i-1], t[i-1])
        y[i] = y[i-1] + 0.5 * h * (f(y[i-1], t[i-1]) + f(y_predict, t[i]))
    return y

# Lorenz system equations
def lorenz_system(y, t, sigma=10, beta=8/3, rho=28):
    x, y, z = y
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

# Function to plot the phase portrait
def plot_phase_portrait(solution, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(solution[:, 0], solution[:, 1], solution[:, 2])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(title)
    plt.show()

# Initial conditions and time points
initial_conditions = np.array([1, 1, 1])
t_lorenz = np.linspace(0, 50, 10000)

# Solve Lorenz system using both methods
lorenz_solution_rk = runge_kutta_4th_order(lambda y, t: lorenz_system(y, t), initial_conditions, t_lorenz)
lorenz_solution_ec = euler_cauchy_method(lambda y, t: lorenz_system(y, t), initial_conditions, t_lorenz)

# Visualize the solutions
plot_phase_portrait(lorenz_solution_rk, "Lorenz System - Runge-Kutta 4th Order")
plot_phase_portrait(lorenz_solution_ec, "Lorenz System - Euler-Cauchy Method")
