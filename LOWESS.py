import numpy as np
import matplotlib.pyplot as plt

# ГЕНЕРАЦИЯ ОБУЧАЮЩЕЙ ВЫБОРКИ 
np.random.seed(1)
N = 100

# Генерируем номера точек от 1 до 100
X = np.arange(1, N + 1)

# Базовая нелинейная функция
y_true = 13.0 + 1.5 * np.sin(X * 0.1) + 1.0 * np.cos(X * 0.05)

# Добавляем обычный случайный шум
noise1 = 0.2
y_range = np.max(y_true) - np.min(y_true)
computed_scale = y_range * noise1

noise = np.random.normal(loc=0.0, scale=computed_scale, size=N)
y_clean = y_true + noise  # Данные без выбросов

# Добавляем ровно 5 крупных случайных выбросов (как на Рисунке 3)
y_dirty = y_clean.copy()
outlier_indices = [6, 71, 76, 69, 86]  # Индексы для демонстрации аномалий
y_dirty[6] = 10.0
y_dirty[71] = 26.8
y_dirty[76] = 18.2
y_dirty[69] = 3.0
y_dirty[86] = 2.0

# 3. АЛГОРИТМ LOWESS
def lowess_1d(X_train, y_train, X_query, beta=0.2, degree=1):
    M_query = X_query.shape[0] # -> хранит int 100
    predictions = np.zeros(M_query)
    
    # beta определяет долю точек (размер окна)
    k_neighbors = int(np.ceil(beta * len(X_train))) # -> int чтобы использовать как индекс, ceil - округление вверх, beta*len(X_train) - 20 соседей
    
    for i in range(M_query):
        x_q = X_query[i]
        
        # Расстояния на одномерной оси
        distances = np.abs(X_train - x_q)
        neighbor_index = np.argsort(distances)[:k_neighbors]
        
        X_neigh = X_train[neighbor_index]
        y_neigh = y_train[neighbor_index]
        dist_neigh = distances[neighbor_index]
        
        max_dist = dist_neigh[-1]
        if max_dist == 0: max_dist = 0.000001

        # Трикубическая функция весов Кливленда
        u = dist_neigh / max_dist
        weights = (1 - u**3)**3
        W = np.diag(weights)
        
        # Формируем матрицу плана в зависимости от степени полинома
        if degree == 1:
            A = np.column_stack((np.ones(k_neighbors), X_neigh))
            a_q = np.array([1, x_q])
        else:  # degree == 2 (Квадратичная)
            A = np.column_stack((np.ones(k_neighbors), X_neigh, X_neigh**2))
            a_q = np.array([1, x_q, x_q**2])
            
        # Решение ВМНК
        ATA = A.T @ W @ A
        ATWy = A.T @ W @ y_neigh
        
        beta_coef, _, _, _ = np.linalg.lstsq(ATA, ATWy, rcond=None)
        predictions[i] = np.dot(a_q, beta_coef)
        
    return predictions

# 1: Сравнение Lowess с разным коэффициентом сглаживания beta
plt.figure(figsize=(11, 6.5))
plt.scatter(X, y_dirty, color='gray', s=45, label='Данные с шумом и выбросами')
plt.plot(X, y_true, 'g--', linewidth=1.8, label='Истинная зависимость')

betas = [0.05, 0.2, 0.5, 0.8]
colors = ['red', 'orange', 'blue', 'purple']

for b, c in zip(betas, colors):
    y_pred = lowess_1d(X, y_dirty, X, beta=b, degree=1)
    plt.plot(X, y_pred, color=c, linewidth=2, label=f'Lowess (β={b})')

plt.title('LOWESS с разными бетта', fontsize=12)
plt.xlabel('Номер точки')
plt.ylabel('Значение Y')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.show()

# 2: Сравнение локально-линейной и полиномиальной регрессии (beta=0.2)
plt.figure(figsize=(11, 6.5))
plt.scatter(X, y_dirty, color='gray', s=45, label='Данные')

y_linear = lowess_1d(X, y_dirty, X, beta=0.2, degree=1)
y_quadratic = lowess_1d(X, y_dirty, X, beta=0.2, degree=2)

plt.plot(X, y_linear, color='blue', linewidth=2, label='Локально-линейная')
plt.plot(X, y_quadratic, color='red', linewidth=2, label='Локально-квадратичная')

plt.title('Локально-линейная и полиномиальная регрессия (бета=0.2)', fontsize=12)
plt.xlabel('Номер точки')
plt.ylabel('Значение Y')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.show()

# 3: Влияние выбросов на Lowess (beta=0.2)
plt.figure(figsize=(11, 6.5))

# Отрисовка фона и чистых точек
clean_mask = np.ones(N, dtype=bool)
clean_mask[outlier_indices] = False
plt.scatter(X[clean_mask], y_clean[clean_mask], color='lightgray', edgecolors='darkgray', s=45, label='Данные без выбросов (фон)')
plt.scatter(X[outlier_indices], y_dirty[outlier_indices], color='red', s=100, zorder=5, label='Выбросы')

# Расчет Lowess для двух выборок
y_pred_clean = lowess_1d(X, y_clean, X, beta=0.2, degree=1)
y_pred_dirty = lowess_1d(X, y_dirty, X, beta=0.2, degree=1)

plt.plot(X, y_pred_clean, color='green', linewidth=2.5, label='Lowess без выбросов')
plt.plot(X, y_pred_dirty, color='purple', linewidth=2, label='Lowess с выбросами')

plt.title('lowess с выбросами и без (бета=0.2)', fontsize=12)
plt.xlabel('Номер точки')
plt.ylabel('Значение Y')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.show()
