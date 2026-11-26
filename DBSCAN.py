import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
# from plot_helper import save_plot
from sklearn.decomposition import PCA
import plotly.express as px

# Считываем файл, сохраняем его в data, удаляем первый столбец, а dropna удалит все пустоты
data = pd.read_csv('CC GENERAL.csv').iloc[:,1:].dropna()

# Стандартизируем данные
data = np.array(data, dtype='float')

min_max_scaler = preprocessing.StandardScaler()
scaled_data = min_max_scaler.fit_transform(data)

# Проведём кластеризацию k-means
kmeans = KMeans(init='k-means++', n_clusters = 3, n_init = 15)
kmeans.fit(scaled_data)

# Проведём кластеризацию DBSCAN при параметрах по умолчанию
clustering = DBSCAN().fit(scaled_data)
labels = clustering.labels_

# 1. Уникальные метки кластеров
unique_labels = np.unique(labels)

# 2. Количество кластеров (исключая шум -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# 3. Процент некластеризованных наблюдений (метка -1)
noise_percent = (np.sum(labels == -1) / len(labels)) * 100

print("Уникальные метки кластеров DBSCAN:", unique_labels)
print(f"Количество кластеров: {n_clusters}")
print(f"Процент некластеризованных наблюдений (шум): {noise_percent:.2f}%")

# Параметры DBSCAN
DBSCAN(
    eps=0.5, # Расстояние от точки до корневой точки чтобы отнести её к кластеру
    min_samples=5, # Минимальное количество объектов в кластере
    metric='euclidean', # Расстояние до точек
    metric_params=None, # Словарь с доп. настройками для выбранной метрики
    algorithm='auto', # Структура данных, с помощью которой алгоритм быстро ищет соседей в пространстве
    leaf_size=30, # Размер минимального узла дерева
    p=None, # Степень для расстояния Минковского. При p = 1 Манх. p = 2 Евк.
    n_jobs=None # Можно поставить -1 для очень больших данных (ядра процессора)
)

# Задаем диапазон значений eps от 0.2 до 5.2 с шагом 0.2 для перебора
eps_values = np.arange(0.2, 5.2, 0.2)

n_clusters_list = []
noise_percent_list = []

# Запускаем DBSCAN для каждого eps
for eps in eps_values:
    db = DBSCAN(eps=eps, n_jobs=-1, metric='euclidean').fit(scaled_data)
    labels=db.labels_ # Список номеров кластеров

    # Считаем количество кластеров исключая шум
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Считаем процентр не кластеризованных наблюдений (метка -1)
    noise_percent = np.abs((np.sum(labels == -1) / len(labels)) * 100)

    n_clusters_list.append(n_clusters)
    noise_percent_list.append(noise_percent)

min_samples_values = range(2, 21)
n_clusters_list_second = []
noise_percent_list_second = []

# Запускаем DBSCAN для каждого min_samples (eps по умолчанию 0.5)
for ms in min_samples_values:
    db = DBSCAN(min_samples=ms, n_jobs=-1).fit(scaled_data)
    labels=db.labels_

    # Считаем количество кластеров (исключая шум -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Считаем процент не кластеризованных наблюдений (метка -1)
    noise_percent = (np.sum(labels == -1) / len(labels)) * 100

    n_clusters_list_second.append(n_clusters)
    noise_percent_list_second.append(noise_percent)


# Строим интерактивные графики в Plotly
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    subplot_titles=(
        "Количество кластеров в зависимости от min_samples",
        "Процент не кластеризованных наблюдений (шума) в зависимости от min_samples"
    )
)

# Первый график: количество кластеров
fig.add_trace(
    go.Scatter(x=list(min_samples_values), y=n_clusters_list_second, mode='lines+markers', name='Количество кластеров'),
    row=1, col=1
)

# Второй график: процент шума
fig.add_trace(
    go.Scatter(x=list(min_samples_values), y=noise_percent_list_second, mode='lines+markers', name='Процент шума (%)', line=dict(color='red')),
    row=2, col=1
)

fig.update_xaxes(title_text="Минимальное количество точек в кластере (min_samples)", row=2, col=1)
fig.update_yaxes(title_text="Число кластеров", row=1, col=1)
fig.update_yaxes(title_text="Шум (%)", row=2, col=1)
fig.update_layout(height=700, title_text="Зависимость результатов DBSCAN от параметра min_samples (eps=0.5)")

# fig.show()
# Строим графики
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    subplot_titles=(
        "Количество кластеров в зависимости от eps",
        "Процент не кластеризованных наблюдений (шума) в зависимости от eps"
    )
)

# Первый график: количество кластеров
fig.add_trace(
    go.Scatter(x=eps_values, y=n_clusters_list, mode='lines+markers', name='Количество кластеров'),
    row=1, col=1
)

# Второй график: процент шума
fig.add_trace(
    go.Scatter(x=eps_values, y=noise_percent_list, mode='lines+markers', name='Процент шума (%)', line=dict(color='red')),
    row=2, col=1
)

fig.update_xaxes(title_text="Максимальная дистанция (eps)", row=2, col=1)
fig.update_yaxes(title_text="Число кластеров", row=1, col=1)
fig.update_yaxes(title_text="Шум (%)", row=2, col=1)
fig.update_layout(height=700, title_text="Зависимость результатов DBSCAN от параметра eps (min_samples=5)")

# fig.show()

# ПУНКТ 6: ПОИСК ОПТИМАЛЬНЫЙ ПАРАМЕТРОВ
eps_range = np.arange(1.2, 4.5, 0.1)
min_samples_range = range(2, 30)

best_params = None
best_labels = None

print("Truth combinations for params: ")
for eps in eps_range:
    eps = round(eps,2)
    for ms in min_samples_range:
        db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit(scaled_data)
        labels = db.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_percent = (np.sum(labels == -1) / len(labels)) * 100

        # Проверяем условие задания: от 5 до 7 кластеров и шум меньше 12%
        if 5 <= n_clusters <= 7 and noise_percent <= 12.0:
            print(f"eps={eps}, min_samples={ms} -> Clusters: {n_clusters}, Noise={noise_percent:.2f}")
            
            # Сохранем первую или лучшую по минимальному шуму комбинацию
            if best_params is None:
                best_params = (eps, ms, n_clusters, noise_percent)
                best_labels = labels
if best_params is None:
    print("Best params is none")
else:
    eps_b, ms_b, n_b, noise_p_b = best_params
    print(f"Was pick params: eps = {eps_b}, min_samples = {ms_b}, Clusters: {n_b}, Percent of noise = {noise_p_b}%")
# ПУНКТ 7: ПОНИЖЕНИЕ РАЗМЕРНОСТИ ДО 2D ЧЕРЕЗ МЕТОД ГЛАВНЫХ КОМПОНЕНТ И ВИЗУАЛИЗАЦИЯ
    pca = PCA(n_components = 2, random_state=1)
    pca_data = pca.fit_transform(scaled_data)

    # Собираем DataFrame для удобной отрисовки
    df_plot = pd.DataFrame({
        'PC1': pca_data[:, 0],
        'PC2': pca_data[:, 1],
        'Cluster': best_labels.astype(str)
    })
    sorted_clusters = sorted(df_plot['Cluster'].unique(), key=lambda x: int(x))
    fig = px.scatter(
            df_plot,
            x='PC1',
            y='PC2',
            color='Cluster',
            category_orders={'Cluster': sorted_clusters},
            color_discrete_map={'-1': 'lightgrey'},  # Шум делаем светло-серым, чтобы не мешал
            title=f'Кластеризация DBSCAN в 2D (PCA) | eps={eps_b}, min_samples={ms_b} | Шум: {noise_p_b:.1f}%',
            labels={'PC1': 'Главная компонента 1 (PC1)', 'PC2': 'Главная компонента 2 (PC2)', 'Cluster': 'Кластер'},
            opacity=0.75
        )

        # Делаем точки чуть компактнее для наглядности
    fig.update_traces(marker=dict(size=5))

    fig.show()