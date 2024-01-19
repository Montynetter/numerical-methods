import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.ndimage import uniform_filter1d, median_filter

# Генерация данных
np.random.seed(42)
x = np.random.rand(100)
y = 2 * x + 1  # истинная линейная зависимость
noise = np.random.normal(0, 0.5, 100)  # шум
y_noisy = y + noise

# Построение диаграммы рассеяния
plt.scatter(x, y_noisy)
plt.title("Диаграмма рассеяния с шумом")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Расчет линейной регрессии
model = LinearRegression().fit(x.reshape(-1, 1), y_noisy)
slope = model.coef_[0]
intercept = model.intercept_

# Вывод параметров регрессии
print(f"Угловой коэффициент: {slope:.2f}, Пересечение: {intercept:.2f}")






#4.3
# Скользящее среднее
window_size = 5
smoothed_data = uniform_filter1d(y_noisy, size=window_size)

# Медианная фильтрация
median_filtered_data = median_filter(y_noisy, size=window_size)

# Визуализация результатов
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x, smoothed_data)
plt.title("Сглаженные данные (скользящее среднее)")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 2, 2)
plt.scatter(x, median_filtered_data)
plt.title("Сглаженные данные (медианная фильтрация)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
