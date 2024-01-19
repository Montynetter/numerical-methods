import numpy as np
import matplotlib.pyplot as plt

# Генерация данных
np.random.seed(42)  # для воспроизводимости результатов

# а) Временной ряд с хаотической динамикой
# Пример: логистическое отображение
def chaotic_time_series(n):
    data = np.zeros(n)
    data[0] = 0.5  # начальное значение
    r = 3.9  # параметр хаоса
    for i in range(1, n):
        data[i] = r * data[i-1] * (1 - data[i-1])
    return data

chaotic_data = chaotic_time_series(1000)

# б) Случайный временной ряд
random_data = np.random.rand(1000)

# в) Нормально распределённые значения
normal_data = np.random.normal(loc=0, scale=1, size=1000)

# Функция для расчета статистических характеристик и гистограммы
def analyze_data(data, title):
    mean = np.mean(data)
    variance = np.var(data)

    plt.hist(data, bins=30, density=True)
    plt.title(f'Гистограмма для {title}\nСреднее: {mean:.2f}, Дисперсия: {variance:.2f}')
    plt.show()

# Анализ каждого набора данных
analyze_data(chaotic_data, "хаотического временного ряда")
analyze_data(random_data, "случайного временного ряда")
analyze_data(normal_data, "нормально распределённых значений")
