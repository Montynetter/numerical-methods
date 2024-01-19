import numpy as np
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

# Linear oscillator equations
def linear_oscillator(y, t):
    y1, y2 = y
    return np.array([y2, -y1])

# Exact solution for the linear oscillator
def exact_solution_linear_oscillator(t, y0):
    return np.array([y0[0] * np.cos(t), -y0[0] * np.sin(t)]).T

# Function to compute the maximum error
def compute_error(numerical_solution, exact_solution):
    return np.max(np.abs(numerical_solution - exact_solution))

# Test the methods and analyze errors and computation times
y0 = np.array([1, 0])  # Initial conditions
step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
results = []

for h in step_sizes:
    t_h = np.arange(0, 10, h)
    exact_solution_h = exact_solution_linear_oscillator(t_h, y0)

    # Runge-Kutta method
    start_time = time.time()
    rk_solution_h = runge_kutta_4th_order(linear_oscillator, y0, t_h)
    rk_time = time.time() - start_time
    rk_error = compute_error(rk_solution_h, exact_solution_h)

    # Euler-Cauchy method
    start_time = time.time()
    ec_solution_h = euler_cauchy_method(linear_oscillator, y0, t_h)
    ec_time = time.time() - start_time
    ec_error = compute_error(ec_solution_h, exact_solution_h)

    results.append((h, rk_error, rk_time, ec_error, ec_time))

# Print results
for h, rk_error, rk_time, ec_error, ec_time in results:
    print(f"Step size: {h}, RK Error: {rk_error}, RK Time: {rk_time}, EC Error: {ec_error}, EC Time: {ec_time}")
