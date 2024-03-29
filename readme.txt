1. Исследование работоспособности численных методов решения системы линейных
алгебраических уравнений (15 баллов)
1.0. Генерация входных данных
В процессе выполнения задания потребуется использование в качестве входных данных
квадратных матриц большой размерности с вещественными элементами. Обратите внимание,
что для тестирования некоторых методов будут необходимы матрицы с определёнными
специфическими свойствами (разреженных, c диагональным преобладанием, с нормой,
лежащей в определённом диапазоне и т. д.).
Пусть A — матрица размера NxN, b – вектор размера N; предусмотрите возможность
получения вектора b, обеспечивающего существование у системы Ax = b известного (заранее
заданного) решения x0.
1.1. Реализация точных и итерационных методов решения СЛАУ (3 балла)
Реализуйте решение СЛАУ методом Гаусса с выбором главного элемента, а также одним из
рассмотренных на лекциях итерационных методов (простой итерации, Якоби, Зейделя).
Сравните полученные результаты и скорость выполнения программ.
1.2. Исследование работоспособности используемых методов для систем различного
размера (4 балла)
Исследуйте зависимость скорости работы обеих программ от размера системы при его
увеличении, а для итерационного метода — также от требуемой точности решения.
1.3. Исследование работоспособности используемых методов для систем с различными
свойствами матрицы коэффициентов. (4 балла)
Изучите особенности работы выбранных методов для систем с различными свойствами
матрицы A. Исследуйте сходимость итерационного метода. Попробуйте оценить
погрешность решения, получаемого методом Гаусса, при увеличении размера системы.
Отличается ли погрешность при использовании стандартного метода Гаусса и метода Гаусса
с выбором главного элемента?
Исследование проводите как для системы с известным решением, так и для системы с
произвольно заданным столбцом свободных членов b.
1.4. Расширенное исследование (4 балла)
Повторите пп. 1.1 — 1.2 с использованием другого языка программирования или среды
разработки (используйте 1-2 дополнительных варианта). Проанализируйте полученные
результаты.
или
Повторите пп. 1.1 — 1.3 для другого итерационного метода. Проанализируйте полученные
результаты


2. Исследование численных методов решения систем обыкновенных дифференциальных
уравнений (15 баллов)
2.1. Численное решение системы ОДУ методами разного порядка точности (10 баллов)
Реализуйте численное решение системы ОДУ размерности N двумя методами: методом 4 (или более
высокого) порядка точности (можно использовать стандартный алгоритм метода Рунге-Кутта 4
порядка) и методом 2 порядка точности (можно использовать одну из двух распространённых
модификаций: метод «предиктор-корректор» (Эйлера-Коши) или модифицированный метод Эйлера).
2.1.а. (5 баллов) Найдите численное решение системы ОДУ, точное решение которой известно
(например, уравнения линейного осциллятора - см. решение в [7] или любом другом учебнике по
теории колебаний). Проведите сравнение погрешностей численного решения и времени вычислений
для методов различного порядка точности. Изучите зависимость величины погрешности от размера
шага.
2.1б. (5 баллов) Найдите численное решение системы ОДУ, точное решение которой неизвестно.
Используйте систему, допускающую существование как сложных (хаотических) режимов поведения,
так и периодических автоколебаний (примеры таких систем можно найти в [4, лек. 5]). Изучите
поведение разницы между решениями, полученными различными методами, для разных режимов
поведения системы - регулярного и хаотического. Как изменение метода интегрирования и шага
интегрирования влияет на вид получаемого фазового портрета?
2.2. Расширенное исследование (5 баллов)
В этом пункте нужно сделать одно из трёх заданий на выбор:
Методы интегрирования с адаптивным шагом
Реализуйте решение системы ОДУ, используя для выбранных ранее методов контроль точности на
шаге. Сравните величину шага, при которой достигается требуемая точность, а также время
вычислений для методов различного порядка точности. Как меняется размер шага интегрирования в
зависимости от поведения траектории системы? Рассмотрите различные примеры: линейный
осциллятор, генератор ван дер Поля в режиме релаксационных автоколебаний, система со сложной
динамикой.
или
Симплектические методы интегрирования
Реализуйте решение гамильтоновой системы ОДУ каким-либо (или несколькими) симплектическими
методами (см. [5], гл. 6). Проверьте работу программы на различных примерах: с известным точным
решением (например, консервативный линейный осциллятор) и со сложной динамикой (например,
система Хенона-Хейлеса, см. [5], ур. (6.14)). Сравните решения и фазовые портреты, получаемые при
использовании аналитического решения (при наличии), симплектического метода, одного из
стандартных методов; в последних двух случаях изучите также влияние шага интегрирования на
получаемый результат. Проанализируйте, как ведут себя решения на больших временах
интегрирования.
или
Интегрирование динамических систем с фиксированным шагом по переменной
Реализуйте интегрирование системы ОДУ с фиксированным шагом по переменной (см. [4, лек. 6.1],
[6]). Проведите оценку погрешности и сравнение полученных решений с решениями, полученными
при использовании стандартных методов интегрирования. Проанализируйте полученные результаты.



Раздел 4. Статистическая обработка эксперимента и регрессионный анализ
Статистическая обработка эксперимента [1-3]
Случайная величина, функция распределения и плотность вероятности. Статистические характеристики: мат. ожидание, дисперсия, среднее квадратичное отклонение, вероятное срединное отклонение. Квантиль уровня величины, медиана. Нормальное распределение.
Многомерная случайная величина и её закон распределения. Двумерный случай: условная плотность распределения и условные характеристики. Функция регрессии. Независимость случайных величин. Корреляционный момент и коэффициент корреляции Пирсона, диаграмма рассеяния. Корреляционная и ковариационная матрицы в многомерном случае. Нормальное распределение векторных величин.
Выборки и выборочные оценки. Гистограммы. Точечные оценки характеристик случайной величины.
Интервальные оценки, доверительная вероятность и доверительный интервал. Стандартные доверительные интервалы нормального распределения.
Проверка статистических гипотез: основные принципы и этапы. Понятия уровня значимости и мощности статистического критерия. Пример применения рассмотренного алгоритма для анализа данных эксперимента.
Регрессионный анализ [1, 3-5]
Фильтрация выборочных данных с целью уменьшения погрешности. Фильтрация аномальных значений.
Задача регрессионного анализа, функция регрессии. Предположения о характере ошибок.
Парная регрессия. Выбор вида функции регрессии. Определение параметров: принцип максимума правдоподобия, методы минимума χ2 и наименьших квадратов.
Задача 4. Обработка набора данных и регрессионный анализ
4.1. (5 баллов) Статистические характеристики сигналов
Для дискретного набора значений переменной получите простейшие статистические характеристики: выборочное среднее, оценки дисперсии, гистограмму (график статистической плотности распределения). Используйте наборы данных, сгенерированные различным образом: а) временной ряд какой-либо системы с хаотической динамикой; б) временной ряд, полученный при помощи генератора случайных чисел; в) набор значений случайной величины, подчиняющейся с той или иной степенью точности нормальному распределению. Проанализируйте полученные результаты.
4.2 (5 баллов) Парная линейная регрессия
Сгенерируйте набор пар точек (y, x), в котором переменные связаны между собой зашумленной линейной зависимостью. Постройте диаграмму рассеяния, определите параметры функции регрессии методом наименьших квадратов. Исследуйте, как получаемый результат зависит от величины шума.
4.3 (5 баллов) Парная нелинейная регрессия
Сгенерируйте набор пар точек (y, x), в котором переменные связаны между собой какой-либо зашумленной нелинейной зависимостью. Постройте диаграмму рассеяния, определите параметры функции регрессии. Исследуйте, как получаемый результат зависит от величины шума и вида зависимости.
или
4.3 (5 баллов) Фильтрация выборочных данных
При выполнении задания 4.2 используйте методы фильтрации выборочных данных (например, методами скользящего среднего и медианной фильтрации). Исследуйте, как получаемый результат зависит от параметров шума и фильтрации.