import numpy as np
import time

# Matrix generation functions
def generate_matrix(N, matrix_type='random', diagonal_dominant=False, sparsity_level=0):
    if matrix_type == 'sparse':
        matrix = np.random.rand(N, N)
        matrix[np.random.rand(*matrix.shape) < sparsity_level] = 0
    else:
        matrix = np.random.rand(N, N)

    if diagonal_dominant:
        np.fill_diagonal(matrix, np.sum(np.abs(matrix), axis=1) + np.random.rand(N))

    return matrix

def generate_diagonally_dominant_matrix(N):
    A = np.random.rand(N, N)
    for i in range(N):
        A[i, i] = sum(np.abs(A[i])) + np.random.rand()
    return A

def generate_vector_b(A, x0):
    return np.dot(A, x0)

# Gaussian elimination function
def gaussian_elimination(A, b):
    N = A.shape[0]
    Ab = np.hstack((A.astype(float), b.reshape(-1, 1).astype(float)))  # Augmented matrix [A | b]
    for i in range(N):
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        for j in range(i + 1, N):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:N], x[i+1:N])) / Ab[i, i]
    return x

# Jacobi method function
def jacobi_method(A, b, tolerance=1e-10, max_iterations=1000):
    N = A.shape[0]
    x = np.zeros(N)
    x_new = np.zeros(N)
    for iteration in range(max_iterations):
        for i in range(N):
            s = sum(A[i, j] * x[j] for j in range(N) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        if np.allclose(x, x_new, atol=tolerance):
            return x_new
        x = np.copy(x_new)
    return x_new

# Comparison and analysis functions
def compare_methods(N, matrix_type='diagonal_dominant', tolerance=1e-10):
    if matrix_type == 'diagonal_dominant':
        A = generate_diagonally_dominant_matrix(N)
    else:
        A = generate_matrix(N, matrix_type)
    x0 = np.random.rand(N)
    b = generate_vector_b(A, x0)
    start_time = time.time()
    x_gaussian = gaussian_elimination(A, b)
    time_gaussian = time.time() - start_time
    start_time = time.time()
    x_jacobi = jacobi_method(A, b, tolerance)
    time_jacobi = time.time() - start_time
    return {
        'Gaussian Solution': x_gaussian,
        'Jacobi Solution': x_jacobi,
        'Time Gaussian': time_gaussian,
        'Time Jacobi': time_jacobi
    }

# Example applications
# 1. Generate a matrix and solve using both methods
N = 10
A = generate_diagonally_dominant_matrix(N)
x0 = np.random.rand(N)
b = generate_vector_b(A, x0)

# Solve using Gaussian elimination
solution_gaussian = gaussian_elimination(A, b)

# Solve using Jacobi method
solution_jacobi = jacobi_method(A, b)

# 2. Compare methods for a specific size
comparison_results = compare_methods(N)

# Display the results
print("Gaussian Solution:", solution_gaussian)
print("Jacobi Solution:", solution_jacobi)
print("Comparison Results:", comparison_results)

