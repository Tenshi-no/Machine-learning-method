import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn import preprocessing
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plot_helper import save_plot

# 1. Загрузка и стандартизация данных
data = pd.read_csv('CC GENERAL.csv').iloc[:, 1:].dropna()
scaler = preprocessing.StandardScaler()
scaled_data = scaler.fit_transform(data)

# ПУНКТ 2: Подбор параметров OPTICS, близких к результатам DBSCAN 

optics_model = OPTICS(
    min_samples=4,
    max_eps=1.7,
    cluster_method='dbscan',
    eps=1.7,
    n_jobs=-1
).fit(scaled_data)

labels_opt = optics_model.labels_
n_clusters_opt = len(set(labels_opt)) - (1 if -1 in labels_opt else 0)
noise_pct_opt = (np.sum(labels_opt == -1) / len(labels_opt)) * 100

print(f"OPTICS (аналог DBSCAN): Кластеров = {n_clusters_opt}, Шум = {noise_pct_opt:.2f}%")

# ПУНКТ 3: Визуализация кластеров (PCA 2D) и график достижимости 

# 2D-визуализация через PCA
pca = PCA(n_components=2, random_state=1)
pca_data = pca.fit_transform(scaled_data)

df_pca = pd.DataFrame({
    'PC1': pca_data[:, 0],
    'PC2': pca_data[:, 1],
    'Cluster': labels_opt.astype(str)
})
sorted_clusters = sorted(df_pca['Cluster'].unique(), key=lambda x: int(x))

fig_2d = px.scatter(
    df_pca,
    x='PC1',
    y='PC2',
    color='Cluster',
    category_orders={'Cluster': sorted_clusters},
    color_discrete_map={'-1': 'lightgrey'},
    title=f'Кластеризация OPTICS в 2D (PCA) | Кластеров: {n_clusters_opt}, Шум: {noise_pct_opt:.1f}%',
    opacity=0.75
)
fig_2d.update_traces(marker=dict(size=5))
save_plot(fig_2d, "optics_pca_2d")
# fig_2d.show()

# График достижимости 
# Для наглядного графика достижимости обучим OPTICS без жесткого обрезания max_eps (или с большим max_eps=5.0),
# чтобы были видны все пики и долины
optics_reach = OPTICS(min_samples=10, max_eps=5.0, xi=0.05, n_jobs=-1).fit(scaled_data)

ordering = optics_reach.ordering_
reachability = optics_reach.reachability_[ordering]
reach_labels = optics_reach.labels_[ordering]

# Заменяем бесконечность (np.inf) на максимальное конечное значение + 0.5 для корректной отрисовки
finite_max = np.max(reachability[np.isfinite(reachability)])
reachability_clean = np.where(np.isinf(reachability), finite_max + 0.5, reachability)

df_reach = pd.DataFrame({
    'Order': np.arange(len(ordering)),
    'Reachability': reachability_clean,
    'Cluster': reach_labels.astype(str)
})

fig_reach = px.bar(
    df_reach,
    x='Order',
    y='Reachability',
    color='Cluster',
    color_discrete_map={'-1': 'lightgrey'},
    title='График достижимости (Reachability Plot) метода OPTICS',
    labels={'Order': 'Порядок обхода точек (ordering_)', 'Reachability': 'Расстояние достижимости (reachability_)'}
)
# Убираем зазоры между столбцами, чтобы график выглядел как непрерывный профиль долин и пиков
fig_reach.update_layout(bargap=0)
save_plot(fig_reach, "optics_reachability_plot")
# fig_reach.show()

# ПУНКТ 4: Исследование работы OPTICS с 5 различными метриками
metrics_list = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'canberra']

print("\n--- Исследование OPTICS на 5 различных метриках (min_samples=10, xi=0.05) ---")
results_metrics = []

for m in metrics_list:
    opt_m = OPTICS(min_samples=5, metric=m, xi=0.01, n_jobs=-1, max_eps=3.0).fit(scaled_data)
    lbl_m = opt_m.labels_
    
    n_cl = len(set(lbl_m)) - (1 if -1 in lbl_m else 0)
    noise_p = (np.sum(lbl_m == -1) / len(lbl_m)) * 100
    
    results_metrics.append({
        'Метрика': m,
        'Количество кластеров': n_cl,
        'Процент шума (%)': round(noise_p, 2)
    })
    print(f"Метрика: {m:10s} | Кластеров: {n_cl:3d} | Шум: {noise_p:6.2f}%")

df_metrics = pd.DataFrame(results_metrics)
print("\nСводная таблица по метрикам:")
print(df_metrics.to_string(index=False))
import plotly.graph_objects as go
import plotly.express as px

fig_reach = go.Figure()

# 1. Сначала рисуем общий контур рельефа с мягкой серой заливкой (чтобы была видна вся кривая и шум)
fig_reach.add_trace(go.Scatter(
    x=df_reach['Order'],
    y=df_reach['Reachability'],
    mode='lines',
    line=dict(color='dimgray', width=1.5),
    fill='tozeroy',
    fillcolor='rgba(150, 150, 150, 0.35)',
    name='Шум (-1) / Общий профиль'
))

# 2. Поверх наносим яркими контрастными цветами точки найденных кластеров (исключая шум -1)
colors = px.colors.qualitative.Dark24  # Яркая контрастная палитра из 24 цветов
clusters = sorted([c for c in df_reach['Cluster'].unique() if c != '-1'], key=int)

for idx, cl in enumerate(clusters):
    sub = df_reach[df_reach['Cluster'] == cl]
    fig_reach.add_trace(go.Scatter(
        x=sub['Order'],
        y=sub['Reachability'],
        mode='markers',
        marker=dict(color=colors[idx % len(colors)], size=5),
        name=f'Кластер {cl}'
    ))

fig_reach.update_layout(
    title='График достижимости (Reachability Plot) метода OPTICS',
    xaxis_title='Порядок обхода точек (ordering_)',
    yaxis_title='Расстояние достижимости (reachability_)',
    template='plotly_white',  # Чистый белый фон с четкой сеткой
    height=600
)

save_plot(fig_reach, "optics_reachability_plot_bright")
fig_reach.show()
