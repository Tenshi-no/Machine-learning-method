import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, mixture
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# 1. Загрузка данных
dataset = fetch_openml(name='quake', version=1, as_frame=True, parser='auto').data

# Находим названия столбцов как обычные строки (берем [0] элемент)
lon_col = [c for c in dataset.columns if 'lon' in c.lower()][0]
lat_col = [c for c in dataset.columns if 'lat' in c.lower()][0]

# --- РИСУНОК 1: Карта очагов по географическим координатам ---
k_demo = 6
X_map = dataset[[lon_col, lat_col]]

# Обучение K-means
kmeans = cluster.KMeans(n_clusters=k_demo, init='random', max_iter=10000, random_state=42)
kmeans_labels = kmeans.fit_predict(X_map) # Находит лучшие координаты для 4 центроидов и присваивает каждой из 2к строк в датасете номер кластера, к которому она оказалась ближе всего
kmeans_centers = kmeans.cluster_centers_ # Забираем из обученной модели координаты найденных центроидов (таблица 4х4)

# Обучение EM-алгоритма
em = mixture.GaussianMixture(n_components=k_demo, max_iter=10000, random_state=42)
em_labels = em.fit_predict(X_map)
em_centers = em.means_

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

# Левый график: K-means
axes1[0].scatter(X_map[lon_col], X_map[lat_col], c=kmeans_labels, s=25, alpha=0.7, cmap='viridis')
axes1[0].scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], c='red', marker='X', s=200, label='Центры')
axes1[0].set_title(f'K-means (K={k_demo})')
axes1[0].set_xlabel('Долгота')
axes1[0].set_ylabel('Широта')
axes1[0].legend(loc='upper left')
axes1[0].grid(True)

# Правый график: EM-алгоритм
axes1[1].scatter(X_map[lon_col], X_map[lat_col], c=em_labels, s=25, alpha=0.7, cmap='viridis')
axes1[1].scatter(em_centers[:, 0], em_centers[:, 1], c='red', marker='X', s=200, label='Центры')
axes1[1].set_title(f'EM-алгоритм (K={k_demo})')
axes1[1].set_xlabel('Долгота')
axes1[1].set_ylabel('Широта')
axes1[1].legend(loc='upper left')
axes1[1].grid(True)

plt.tight_layout()
plt.show()

# --- РИСУНОК 2: Графики показателей для определения оптимального K ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(dataset)

# Задаем список значений К, которые будем проверять
k_range = range(1, 16)
# Здесь будет храниться сумма квадратов расстояний на каждом шаге цикла
inertia_list = []

# Список значений К от 2 до 15 для второго графика. Начинается с 2 так как т.к. минимум необходимо два кластера
ratio_k_range = range(2, 16)
# Пустой список для хранения отношения внутрикластерного расстояния к внекластерному
ratio_list = []

for k in k_range:
    model = cluster.KMeans(n_clusters=k, init='random', max_iter=10000, random_state=42)
    clusterobj = model.fit(X_scaled)
    inertia_list.append(clusterobj.inertia_)
    
    if k >= 2:
        labels = clusterobj.labels_
        centers = clusterobj.cluster_centers_
        # Считаем среднее внутрикластерное расстояние (для каждого из 2к землетрясений находит расстояние по прямой до центра его кластера, а затем берет ср. знач по всем точкам.)
        intra_dist = np.mean([np.linalg.norm(X_scaled[i] - centers[labels[i]]) for i in range(len(X_scaled))])
        # Считаем расстояние между парой каждой парой центроидов
        inter_dist = np.mean(pairwise_distances(centers)[np.triu_indices(k, k=1)])
        # Получаем метрику компактности (насколько плотно точки сгруппированы внутри своих кластеров по сравнению с тем, насколько далеко сами кластеры находятся друг от друга)
        ratio_list.append(intra_dist / inter_dist)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

axes2[0].plot(k_range, inertia_list, marker='o', color='blue', linewidth=2)
axes2[0].set_title('Сумма квадратов расстояний (Метод локтя)')
axes2[0].set_xlabel('Количество кластеров (K)')
axes2[0].set_ylabel('Inertia')
axes2[0].grid(True)

axes2[1].plot(ratio_k_range, ratio_list, marker='o', color='green', linewidth=2)
axes2[1].set_title('Отношение внутрикластерного расстояния к внекластерному')
axes2[1].set_xlabel('Количество кластеров (K)')
axes2[1].set_ylabel('Отношение расстояний')
axes2[1].grid(True)

plt.tight_layout()
plt.show()
