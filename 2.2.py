import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import time

# Runge-Kutta 4th order method with adaptive step size control
def runge_kutta_4th_order_adaptive(f, y0, t_span, tol=1e-6):
    sol = solve_ivp(f, t_span, y0, method='RK45', rtol=tol, atol=tol)
    return sol.t, sol.y.T

# Linear oscillator system
def linear_oscillator(t, y):
    return [y[1], -y[0]]

# Van der Pol oscillator system
def van_der_pol_oscillator(t, y, mu=1000):
    return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

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

# Function to plot step size distribution
def plot_step_size_distribution(step_sizes, title):
    plt.figure(figsize=(10, 4))
    plt.plot(step_sizes, marker='o')
    plt.xlabel('Step number')
    plt.ylabel('Step size')
    plt.title(title)
    plt.show()

# Initial conditions and time spans for different systems
y0_linear = [1, 0]
y0_vdp = [2, 0]
y0_lorenz = [1, 1, 1]
t_span_linear = (0, 10)
t_span_vdp = (0, 3000)  # Extended time for stiff behavior
t_span_lorenz = (0, 50)

# Solving and visualizing different systems
# Linear Oscillator
t_linear, y_linear = runge_kutta_4th_order_adaptive(linear_oscillator, y0_linear, t_span_linear)
plot_step_size_distribution(np.diff(t_linear), "Linear Oscillator - Step Size Distribution")

# Van der Pol Oscillator
t_vdp, y_vdp = runge_kutta_4th_order_adaptive(lambda t, y: van_der_pol_oscillator(t, y), y0_vdp, t_span_vdp)
plot_step_size_distribution(np.diff(t_vdp), "Van der Pol Oscillator - Step Size Distribution")

# Lorenz System
t_lorenz, y_lorenz = runge_kutta_4th_order_adaptive(lambda t, y: lorenz_system(y, t), y0_lorenz, t_span_lorenz)
plot_phase_portrait(y_lorenz, "Lorenz System - Adaptive Runge-Kutta 4th Order")
