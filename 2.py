import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
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

# Testing the methods with different systems
# Initial conditions and time points
y0_linear = [1, 0]
t_linear = np.linspace(0, 10, 100)
y0_vdp = [2, 0]
t_vdp = np.linspace(0, 3000, 30000)  # Extended time for stiff behavior
y0_lorenz = [1, 1, 1]
t_span_lorenz = (0, 50)

# Runge-Kutta and Euler-Cauchy solutions for Linear Oscillator
rk_linear = runge_kutta_4th_order(linear_oscillator, y0_linear, t_linear)
ec_linear = euler_cauchy_method(linear_oscillator, y0_linear, t_linear)

# Adaptive Runge-Kutta for Van der Pol and Lorenz systems
_, y_vdp_adaptive = runge_kutta_4th_order_adaptive(lambda t, y: van_der_pol_oscillator(t, y), y0_vdp, (0, 3000))
_, y_lorenz_adaptive = runge_kutta_4th_order_adaptive(lambda t, y: lorenz_system(y, t), y0_lorenz, t_span_lorenz)

# Visualize results
plot_step_size_distribution(np.diff(t_linear), "Linear Oscillator - Step Size Distribution (Fixed Step)")
plot_phase_portrait(y_vdp_adaptive, "Van der Pol Oscillator - Adaptive Runge-Kutta")
plot_phase_portrait(y_lorenz_adaptive, "Lorenz System - Adaptive Runge-Kutta")
