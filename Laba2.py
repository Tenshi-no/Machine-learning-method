import numpy as np
import pandas as pd
from statsmodels.nonparametric import kernels
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.pyplot as plt

# Гауссовское ядро
def gaussian_kernel(u):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
# Треугольное ядро
def triangular_kernel(u):
    u = np.abs(u)
    return np.where(u <= 1, 1 - u, 0)
# Трикубическое ядро
def tricube_kernel(u):
    u = np.abs(u)
    return np.where(u < 1, (1 - u**3)**3, 0)

# Словарь ядер для удобного перебора
kernel_functions = {
    'gaussian': gaussian_kernel,
    'triangular': triangular_kernel,
    'tricube': tricube_kernel
}

def nadaraya_watson_predict(X_train, y_train, X_test, kernel_func, bandwidth):
    predictions = []
    
    for x_test in X_test:
        # Вычисляем расстояния от тестовой точки до всех обучающих
        # Используем евклидово расстояние, нормированное на bandwidth
        distances = np.sqrt(np.sum(((X_train - x_test) / bandwidth)**2, axis=1))
        
        # Вычисляем веса через ядро
        weights = kernel_func(distances)
        
        # Предсказание - взвешенное среднее
        if np.sum(weights) > 0:
            prediction = np.sum(weights * y_train) / np.sum(weights)
        else:
            prediction = np.mean(y_train)  # берем среднее, так как это самое безопасное предсказание
        
        predictions.append(prediction)
    
    return np.array(predictions)

def select_bandwidth(X_train):
   
    n = len(X_train) # Количество объектов в выборке
    d = X_train.shape[1]  # размерность
    const_silverman = 1.06
    
    # Правило Сильвермана для многомерного случая
    std = np.std(X_train, axis=0)
    bandwidth = const_silverman * std * (n ** (-1 / (d + 4)))
    
    return bandwidth

# Фиксируем рандомное значение, а не получаем каждый раз новое
np.random.seed(1) 

# Объём выборки
num_samples = 2000

# Генерируем входые признаки (равномерное распределение)
x1 = np.random.uniform(low=-2.0, high=2.0, size=num_samples)
x2 = np.random.uniform(low=-1.0, high=2.0, size=num_samples)
x3 = np.random.uniform(low=0.0, high=2*np.pi, size= num_samples)

# Нелинейная зависимость y = f(x1, x2, x3)
# 1. Квадратичная
# 2. Экспоненциальная
# 3. Тригонометрическая
y_clean = (x1*x1) + np.exp(x2) - 4 *np.sin(x3)

# Собираем данные в датафрейм
dataset = pd.DataFrame({
        'x1_quadratic': x1,# Квадратичный признак
        'x2_exponential': x2, # Экспненциальный признак
        'x3_sinusoidal': x3, # Синусоидальный признак (sin(x))
        'target_y': y_clean # Целевая переменная с шумом
    })

print("Первые 5 строк нелинейной выборки")
print(dataset.head())
#-------------------------------------ЗАДАНИЕ 2----------------------------
# Среднее значение шума = 1.0, стандартное отклонение 0.15 (рпазброс +-15%
mult_noise = np.random.normal(loc=1.0, scale=0.15, size=num_samples)
# Мультипликативная помеха
y_with_noise = y_clean * mult_noise

dataset_noise = pd.DataFrame({
        'x1_quadratic': x1, 
        'x2_exponential': x2, 
        'x3_sinusoidal': x3, 
        'target_y': y_with_noise
    })

print("Первые 5 строк нелинейной выборки с помехами")
print(dataset_noise.head())

#-------------------------------------ЗАДАНИЕ 3----------------------------
# Матрица признаков Х и целевая переменная У
x = np.stack([x1, x2, x3], axis = 1) #если поставить 0, то будет построчная склейка
y = y_with_noise

# Разделяем на обучающую и тестовую выборку
x_train, x_test, y_train, y_test, y_clean_train, y_clean_test = train_test_split(x,y,y_clean, test_size=0.2, random_state=1)

# Обучаем модель Надарая-Ватсона
# var_type='ccc' означает, что все 3 наших входных признака — непрерывные
# Алгоритм автоматически подберет оптимальную ширину окна  для каждой переменной
kr = KernelReg(endog=y_train, exog=x_train, var_type='ccc')

# Получаем предсказания на тестовой выборке
# Метод возвращает предсказание и предельные эффекты
y_pred, _ = kr.fit(x_test)

print(f"score относительно зашумленных данных: {r2_score(y_test, y_pred)}")
print(f"score относительно истинной чистой формулы: {r2_score(y_clean_test, y_pred)}")
print('='*70)

print("Сравнение результатов (Надарая-Ватсон):")
for i in range(5):
    print(f"Объект {i+1}: Истинный y = {y_clean_test[i]} | "
          f"Искаженный y = {y_test[i]} | "
          f"Предсказание Надарая-Ватсона = {y_pred[i]}")

#-------------------------------------ЗАДАНИЕ 4----------------------------
# Расчет классической среднеквадратичной ошибки
mse_clean = mean_squared_error(y_clean_test, y_pred)
mse_noise = mean_squared_error(y_test, y_pred)

# Расчет нормированной ошибки E
# Находим размах (y_max - y_min) на тестовой выборке
y_max = y_test.max()
y_min = y_test.min()
range_y = y_max - y_min

# Считаем сумму абсолютных отклонений (y_i - y_pred_i) (ищем ошибки модели)
errors_sum = np.sum(np.abs(y_test - y_pred))

s = len(y_test)
# Формула из задания
E_noisy = (1 / s) * (errors_sum / range_y) 

# Аналогично считаем Е относительно истинной чистой формулы
errors_clean_sum = np.sum(np.abs(y_clean_test - y_pred))
y_clean_max = y_clean_test.max()
y_clean_min = y_clean_test.min()
E_clean = (1 / s) * (errors_clean_sum / (y_clean_max - y_clean_min))

print("ЗАДАНИЕ 4: ОЦЕНКА АДЕКВАТНОСТИ МОДЕЛИ")
print(f"MSE (относительно зашумленных данных): {mse_noise}")
print(f"MSE (относительно чистой формулы): {mse_clean}")
print("-"* 60)
print(f"Ошибка E (относительно зашумленных данных):  {E_noisy}")
print(f"Ошибка E (относительно чистой формулы): {E_clean}")

#-------------------------------------ЗАДАНИЕ 5----------------------------
# Задаем сетку возможных значений параметра размытости h
# 1. start — начало отсчета (первое число в массиве).
# 2. stop — конец отсчета 
# 3. num — сколько всего чисел (точек) нужно сгенерировать отрезке
h_grid = np.linspace(0.05, 1.0, 20)

plt.figure(figsize=(12, 6))
optimal_h_result = {}

# Перебираем каждое ядро
for kernel_name, kernel_func in kernel_functions.items():
    crit_value = []
    
    for h in h_grid:
        y_pred_fixed = nadaraya_watson_predict(x_train, y_train, x_test, kernel_func, h)
        mse_h = mean_squared_error(y_test, y_pred_fixed)
        crit_value.append(mse_h)
    
    # Определяем наилучший h для текущего ядра
    best_h_idx = np.argmin(crit_value) # Использую argmin чтобы найти индекс минимального значения
    best_h = h_grid[best_h_idx] # Сохраняю в переменную именно значение
    best_mse = crit_value[best_h_idx] # Минимальная оценка
    
    # Сохраняем результат
    optimal_h_result[kernel_name] = (best_h, best_mse)
    
    plt.plot(h_grid, crit_value, marker='o', linestyle='-', linewidth=2, 
             label=f'{kernel_name} (min MSE={best_mse:.4f} при h={best_h:.2f})')
    plt.axvline(best_h, color='red', linestyle=':', alpha=0.7)

plt.title('Задание 5: Зависимость MSE от параметра размытости (h) для разных ядер', fontsize=13)
plt.xlabel('Параметр размытости (Bandwidth, h)', fontsize=11)
plt.ylabel('Значение критерия (MSE на тесте)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()

# Вывод сводной таблицы
print("="*70)
print("ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ РАЗМЫТОСТИ (h) ДЛЯ РАЗНЫХ ЯДЕР:")
print("-"*70)
for k_name, (h_opt, mse_opt) in optimal_h_result.items():
    print(f"Ядро: {k_name:15s} | Оптимальный h = {h_opt:.2f} | Минимальный MSE = {mse_opt:.4f}")
print("="*70)  

#-------------------------------------ЗАДАНИЕ 6----------------------------
h_values = [0.30, 1.00]

colors = ['green', 'blue']

num_plot_points = 60 # Количество точек
points_indices = np.arange(1, num_plot_points + 1)
actual_noisy = y_test[:num_plot_points]

# Создаём график
plt.figure(figsize=(14, 7))

# Рисуем реальные зашумлённые точки (фон)
plt.scatter(points_indices, actual_noisy, 
            color='gray', alpha=0.4, edgecolors='k', 
            s=30, label='Реальные данные (с шумом)', zorder=1)

# Для каждого значения h обучаем модель и строим предсказания
results = []  # Сохраняем результаты для вывода метрик

for h, color in zip(h_values, colors):
    # Создаём модель ядерной регрессии
    # bandwidth=[h, h, h] - одинаковая ширина для всех 3 признаков
    kr = KernelReg(endog=y_train, 
                   exog=x_train, 
                   var_type='ccc',
                   bw=[h, h, h])  # <-- здесь задаём h
    
    # Делаем предсказания на тестовой выборке
    y_pred_h, _ = kr.fit(x_test)
    
    # Сохраняем метрики
    mse_clean = mean_squared_error(y_clean_test, y_pred_h)
    mse_noisy = mean_squared_error(y_test, y_pred_h)
    r2_clean = r2_score(y_clean_test, y_pred_h)
    r2_noisy = r2_score(y_test, y_pred_h)
    
    results.append({
        'h': h,
        'mse_clean': mse_clean,
        'mse_noisy': mse_noisy,
        'r2_clean': r2_clean,
        'r2_noisy': r2_noisy
    })
    
    # Рисуем линию предсказаний
    plt.plot(points_indices, y_pred_h[:num_plot_points], 
             color=color, 
             linewidth=2, 
             label=f'h = {h:.2f} (R² чистые={r2_clean:.3f})',
             zorder=2)

# Оформляем график
plt.title('Задание 6: Сравнение оценок регрессии при разных значениях h', fontsize=13, fontweight='bold')
plt.xlabel('Номер точки (Индекс наблюдения)', fontsize=11)
plt.ylabel('Значение зависимой переменной Y', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.show()

#-------------------------------------ЗАДАНИЕ 7----------------------------
y_final_pred, _ = kr.fit(x_test)
y_test_arr = np.array(y_test)
y_clean_test_arr = np.array(y_clean_test)

# Считаем итоговые метрики адекватности
final_mse_noise = mean_squared_error(y_test_arr, y_final_pred)
final_mse_clean = mean_squared_error(y_clean_test_arr, y_final_pred)

# Расчет нормированной ошибки для зашумленного таргета
range_y = y_test_arr.max() - y_test_arr.min()
E_noise = (1 / len(y_test_arr)) * (np.sum(np.abs(y_test_arr - y_final_pred)) / range_y)

# Расчет нормированной ошибки E для чистой формулы
range_y_clean = y_clean_test_arr.max() - y_clean_test_arr.min()
E_clean = (1 / len(y_clean_test_arr)) * (np.sum(np.abs(y_clean_test_arr - y_final_pred)) / range_y_clean)

print(f"Количество объектов на тесте: {len(y_test_arr)}")
print(f"Финальный MSE (относительно шума):  {final_mse_noise}")
print(f"Финальный MSE (относительно закона): {final_mse_clean}")
print(f"Финальная ошибка E (относительно шума):  {E_noisy}")
print(f"Финальная ошибка E (относительно закона): {E_clean}")
print("-" * 50)

# 4. Проверяем локальную точность на случайных трех объектах из теста
print("Выборочная проверка предсказаний для трех случайных объектов:")
indices = [15, 30, 70] 
for idx in indices:
    print(f"Объект №{idx}:")
    print(f"Реальное зашумленное значение Y: {y_test_arr[idx]}")
    print(f"Чистое теоретическое значение Y: {y_clean_test_arr[idx]}")
    print(f"Восстановленное моделью значение: {y_final_pred[idx]}")
    print(f"Абсолютная ошибка восстановления закона: {np.abs(y_clean_test_arr[idx] - y_final_pred[idx])}")
    
#-------------------------------------ЗАДАНИЕ 8----------------------------
np.random.seed(1)

sample_sizes = [200, 500, 1000, 2000]
research_results = []

for size in sample_sizes:
    # Генерация данных
    x1_res = np.random.uniform(low=-2.0, high=2.0, size=size)
    x2_res = np.random.uniform(low=-1.0, high=2.0, size=size)
    x3_res = np.random.uniform(low=0.0, high=2*np.pi, size=size)
    
    y_clean_res = (x1_res**2) + np.exp(x2_res) - 4 * np.sin(x3_res)
    mult_noise_res = np.random.normal(loc=1.0, scale=0.15, size=size)
    y_noisy_res = y_clean_res * mult_noise_res
    
    X_res = np.stack([x1_res, x2_res, x3_res], axis=1)
    
    X_tr, X_te, y_tr, y_te, y_c_tr, y_c_te = train_test_split(
        X_res, y_noisy_res, y_clean_res, test_size=0.2, random_state=1
    )
    
    # Перебираем ядра
    for kernel_name, kernel_func in kernel_functions.items():
        # Выбираем bandwidth
        bandwidth = select_bandwidth(X_tr, y_tr, kernel_func)
        
        # Предсказание
        y_pred_res = nadaraya_watson_predict(X_tr, y_tr, X_te, kernel_func, bandwidth)
        
        # Метрики
        y_c_te_arr = np.array(y_c_te)
        r2_clean = r2_score(y_c_te_arr, y_pred_res)
        
        range_y_c = y_c_te_arr.max() - y_c_te_arr.min()
        E_clean = (1 / len(y_c_te_arr)) * (np.sum(np.abs(y_c_te_arr - y_pred_res)) / range_y_c)
        
        research_results.append({
            'Объем выборки': size,
            'Тип ядра': kernel_name,
            'R^2 (Закон)': round(r2_clean, 4),
            'Ошибка E (%)': round(E_clean * 100, 2)
        })
        
        print(f"Выборка: {size:4d} | Ядро: {kernel_name:10s} | "
              f"Bandwidth: {bandwidth.mean():.3f} | "
              f"R^2: {r2_clean:.4f}, E: {E_clean*100:.2f}%")
