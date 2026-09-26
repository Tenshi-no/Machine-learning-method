import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. ГЕНЕРАЦИЯ ДАТАСЕТА
np.random.seed(42)
n = 500
data = {
    "age": np.random.normal(35, 12, n),
    "income": np.random.lognormal(10, 0.5, n),
    "years_experience": np.random.uniform(0, 30, n),
    "education": np.random.choice(["School", "Bachelor", "Master", "PhD"], n),
    "department": np.random.choice(["Sales", "IT", "HR", "Marketing"], n),
    "is_churn": np.random.binomial(1, 0.2, n)
}

df = pd.DataFrame(data)

# Создаем искусственные проблемы
df.loc[np.random.choice(df.index, 20), "age"] = np.nan
df.loc[np.random.choice(df.index, 15), "income"] = np.nan
df.loc[np.random.choice(df.index, 10), "years_experience"] = -5
df.loc[np.random.choice(df.index, 8), "education"] = "Unknown"
df.loc[np.random.choice(df.index, 5), "department"] = None

# Добавим немного пропусков в целевую переменную (чтобы было что удалять по заданию)
df.loc[np.random.choice(df.index, 12), "is_churn"] = np.nan 

df.loc[100, "income"] = 1_000_000 # Выброс
df.to_csv("synthetic_ml_data.csv", index=False)


# pd.set_option('display.max_columns', None) # Используем для того, чтоб вывелась вся таблица в ширину
# pd.set_option('display.width', 1000) # Используем для того, чтоб вывелась вся таблица в ширину

print(df.head(10))

print("Статистика по числовым признакам")
print(df.describe())

print("Уникальные значения и частоты для категориальных признаков")
# Ниже отбираются категориальные признаки (текстовые колонки)

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    print(f"Частота значений столбцов {col}:")
    print(df[col].value_counts()) # Автоматически находит текстовые столбцы и для каждого из них выводит список уникальных слов и сколько раз они встретились
    print("-"*50)
    
print("Доля пропусков по каждому столбцу")
miss = df.isnull().mean()*100
print(round(miss))

# --- 2. ВЫПОЛНЕНИЕ ЗАДАНИЙ ПО ОЧИСТКЕ ДАННЫХ ---
print("ОЧИСТКА ДАННЫХ")

# Задание: Удалите строки, где целевая переменная (is_churn) отсутствует
churn_null_pct_before = round(df['is_churn'].isnull().mean() * 100, 2)
print(f"Процент пропусков в is_churn ДО удаления: {churn_null_pct_before}%")

# Задание: Удалите строки, где целевая переменная (is_churn) отсутствует
df = df.dropna(subset=['is_churn'])

churn_null_pct_after = round(df['is_churn'].isnull().mean() * 100, 2)
print(f"Процент пропусков в is_churn ПОСЛЕ удаления: {churn_null_pct_after}%")
print("-" * 40)


# Задание: Исправьте некорректные значения (замените отрицательные значения years_experience на NaN)
negative_exp_count = (df['years_experience'] < 0).sum()
print(f"Количество строк с отрицательным опытом ДО замены: {negative_exp_count}")

# Производим замену отрицательных значений на NaN
df.loc[df['years_experience'] < 0, 'years_experience'] = np.nan

negative_exp_count_after = (df['years_experience'] < 0).sum()
print(f"Количество строк с отрицательным опытом ПОСЛЕ замены: {negative_exp_count_after}")

# Считаем итоговый процент пропусков в опыте работы
exp_null_pct = round(df['years_experience'].isnull().mean() * 100, 2)
print(f"Общий процент пропусков в 'years_experience': {exp_null_pct}%")
print("-" * 40)


# Итоговый вывод пропусков по всему датасету в процентах
print("Итоговый процент пропусков по всем столбцам:")
print(round(df.isnull().mean() * 100, 2).astype(str) + '%')

print("Заполните пропуски")

#Находим все числовые столбцы кроме is_churn
numeric_cols = df.select_dtypes(include=['number']).columns
if 'is_churn' in numeric_cols:
    numeric_cols = numeric_cols.drop('is_churn')
    
print("Заполнение числовых столбцов медианой",list(numeric_cols) )


for col in numeric_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)
    
# Заполнение категориальных признаков. Находим все текстовые столбцы
categorical_cols = df.select_dtypes(include=['object']).columns
print("Заполнение категориальных столбцов модой: ", list(categorical_cols))

#.mode - возвращает самое частое слово в строке
for col in categorical_cols:
    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(mode_value) #fillna заполняет пропуски, пробегая по указанному столбцу. находит ячейки, где написано NaN и вставляет туда значение, которое мы передали как параметр
print("Пороцент пропусков в датасете после заполнения: ")
print(round(df.isnull().mean()*100, 2).astype(str) + "%")


print("ОБРАБОТКА ВЫБРОСОВ")
print('-'*50)
print("Определение выбросов")
#Когда мы задаем кванитль 0.25, мы просим найти человека, который стоит ровно на 25%
#Когда задаем 0.75, то находим человека, стоящего на границе 75%

Q1 = df['income'].quantile(0.25)
Q3 = df["income"].quantile(0.75)

IQR = Q3 - Q1 #Вычесляем интервальный размах
#Определяем границы для нормальных значений
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q1 + 1.5 * IQR

print(f"Нижняя граница нормы: {round(lower_bound, 2)}")
print(f"Высшая граница нормы: {round(upper_bound, 2)}")

print("Сравнение статистики")
# Чтобы показать статистику до, надо взять исходный массив income
income_before = pd.Series(np.random.lognormal(10, 0.5, n))
income_before.loc[100] = 1000000 #самый большой выброс

print("Метрики ДО")

stats_before = {
    "Минимум": income_before.min(),
    "Максимум": income_before.max(),
    "Среднее": income_before.mean(),
    "Медиана": income_before.median()
}
stats_after = {
    "Минимум": df['income'].min(),
    "Максимум": df['income'].max(),
    "Среднее": df['income'].mean(),
    "Медиана": df['income'].median()
}

comparison_df = pd.DataFrame({"ДО обработки": stats_before, "ПОСЛЕ обработки": stats_after})

# Округляем для читаемости
print(comparison_df.round(2))
print("-" * 40)

print("Кодирование категориальных признаков")

#Смотрим на типы данных всех столбцов в датасете
print("Типы данных всех столбцов: ")
print(df.dtypes)
print("-"*30)

#Автоматически находим текст
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Категориальные признаки: ")
print(categorical_cols)
print("-"*50)

print("Для признака education применяем OrdinalEncoder")
# Задаем четкий список категорий в нужном нам порядке
education_order = [["Unknown", "School", "Bachelor", "Master", "PhD"]]

# Создаем энкодер и передаем ему наш заданный порядок через параметр categories
encoder = OrdinalEncoder(categories = education_order)
# Обучаем кодировщик и трансформируем столбец education
df['education_encoded'] = encoder.fit_transform(df[['education']])

print("Результат кодирования: ")
print(df[['education', 'education_encoded']].drop_duplicates().sort_values(by='education_encoded'))#комбинация методов используется для того, чтобы оставить в таблице только уникальные строки и расставить по порядку
print("-"*50)

print("УНИТАРНОЕ КОДИРОВАНИЕ")
# Создаем кодировщик.
# sparse_output=False возвращает обычный массив (не разреженную матрицу), чтобы его легко превратить в таблицу.
# drop='first' можно убрать, но он часто используется, чтобы избежать мультиколлинеарности.
encoder_ohe = OneHotEncoder(sparse_output = False, dtype=int)
#Обучаем и кодируем столбец "department" для расстановки нулей и единиц
ohe_encoded = encoder_ohe.fit_transform(df[['department']])
#Получаем названия для новых столбцов
col_names = encoder_ohe.get_feature_names_out(['department'])
#Создаем мини-таблицу из закодированных  данных и объединяекм с основным датасетом
df_ohe = pd.DataFrame(ohe_encoded, columns=col_names, index=df.index)
df = pd.concat([df, df_ohe], axis=1)
#Выведем первые 5 строк исходного столбца и новых закодированных колонок
print("Результат унитарного кодирования")
print(df[['department'] + list(col_names)].head(5))
print("-"*40)

print("МАСШТАБИРОВАНИЕ ПРИЗНАКОВ")
print("Выделение числовых признаков")
#Находим все числовые столбцы
all_numeric = df.select_dtypes(include=['number'])
#Если есть целевая переменная, то исключаем её
if "is_churn" in all_numeric.columns:
    X_numeric = all_numeric.drop(columns=['is_churn'])
else:
    X_numeric = all_numeric
#Выводим список всех получившихся признаков
print("Итоговый список числовых признаков для модели: ")
print(list(X_numeric.columns))
print("-" * 40)
#Проверка
print("Первые 5 строк числового датафрейма: ")
print(X_numeric.head(5))

print("Разделение на обучающую и тестовую выборку")

# 1.  признаки (X), целевая переменная (y)
# В X берем только числовые признаки, которые выделили на прошлом шаге
X = X_numeric
y = df['is_churn']

#Разделяем в пропорции 80 на 20 и фиксируем случайность, чтобы при каждом запуске выборки были одинаковыми
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)


#  Выводим размеры получившихся выборок
print(f"Размер обучающей выборки признаков (X_train): {X_train.shape}")
print(f"Размер тестовой выборки признаков (X_test):  {X_test.shape}")
print(f"Размер обучающих ответов (y_train): {y_train.shape}")
print(f"Размер тестовых ответов (y_test):   {y_test.shape}")
print("-" * 60)

print("Масштабирование признаков (Standard scaler)")

#создание объекта скалера
scaler = StandardScaler()

#Обучение скалера только на обучающей выборке и сразу её трансформируем
# fit_transform вычисляет среднее и среднеквадратичное отклонение для каждого столбца в X_train и масштабирует их
X_train_scaled = scaler.fit_transform(X_train)

# трансформируем тестовую выборку тем же самым скалером (БЕЗ переобучения)
# Используем метод transform(), чтобы применить среднее и среднеквадратичное отклонение, посчитанные на X_train
X_test_scaled = scaler.transform(X_test)

#Превращаем результат обратно в DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns = X_train.columns, index = X_train.index)

print("Проверка масштабирования на обучающей выборке (первые 3 колонки):")
print("Средние значения (должны быть около 0):")
print(X_train_scaled_df.iloc[:, :3].mean().round(2))
print("\nСтандартное отклонение (должно быть равно 1):")
print(X_train_scaled_df.iloc[:, :3].std().round(2))
print("-" * 40)

print("СРАВНЕНИЕ СТАТИСТИКИ ДО И ПОСЛЕ МАСШТАБИРОВАНИЯ")

#Выбираем первые 4 признака для демонстрации
demo_cols = ['age', 'income', 'years_experience', 'education_encoded']

#Считаем метрики до масштабирования
mean_before = X_train[demo_cols].mean()
std_before = X_train[demo_cols].std()
#Cчитаем метрики после масштабирования
mean_after = X_train_scaled_df[demo_cols].mean()
std_after = X_train_scaled_df[demo_cols].std()
#Ниже объединение в таблицу
scaling_table = pd.DataFrame({
    "Среднее до": mean_before,
    "Среднее после": mean_after,
    "Разброс ср. откл. до": std_before,
    "Разброс ср. откл. после": std_after
})
print(scaling_table.round(2))
print('-'*50)

print("РЕАЛИЗАЦИЯ НОВЫХ ПРИЗНАКОВ (feature engineering)")
# Признак 1: Income per year experience
# income / years_experience, только если years_experience > 0
df['income_per_year_exp'] = np.where(
    df['years_experience'] > 0, df['income'] / df['years_experience'], 0 #Если опыт 0 или отрицательный, то ставим 0
)
df['income_per_year_exp'] = df['income_per_year_exp'].round(2)
print("Признак income_per_year_exp создан")

# Признак 2: бинарный high_income
#1 если доход выше медианы, иначе 0
income_median = df['income'].median()
df["high_income"] = (df['income'] > income_median).astype(int)
print(f"Признак high_income создан (медиана дохода = {income_median:.2f})")

print("-"*50)

print("Задание 7.  ОЦЕНКА ПЛЛОТНОСТЕЙ ВЕРОЯТНОСТИ ПРИЗНАКОВ")

# 1. Гистограммы + KDE до логарифмирования

print("-"*50)
print("Задание 8. ОЦЕНКА ЗНАЧИМОСТИ ПРИЗНАКОВ")
print("1. Гистограммы и KDE для ключевых числовых признаков")
# Выбираем признаки для анализа
features_to_analyze  = ['age', 'income', 'years_experience', 'income_per_year_exp']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feat in enumerate(features_to_analyze):
    sns.histplot(
        df[feat],
        kde=True,           # ядерная оценка плотности
        bins=30,            # количество бинов
        color='steelblue',
        ax=axes[idx]
    )
    axes[idx].set_title(f'Распределение {feat}', fontsize=12)
    axes[idx].set_xlabel(feat)
    axes[idx].set_ylabel('Частота')

plt.tight_layout()
plt.show()


# 2. ОЦЕНКА АСИММЕТРИИ И ЭКСЦЕССА ДО ПРЕОБРАЗОВАНИЯ

print("\n2. Асимметрия (skew) и эксцесс (kurtosis) ДО логарифмирования")
print("-" * 60)

skew_before = {}
kurt_before = {}

for feat in features_to_analyze:
    skew_before[feat] = stats.skew(df[feat])
    kurt_before[feat] = stats.kurtosis(df[feat])  # excess kurtosis (нормальное = 0)

stats_before_df = pd.DataFrame({
    'Признак': features_to_analyze,
    'Skew (асимметрия)': [round(skew_before[f], 4) for f in features_to_analyze],
    'Kurtosis (эксцесс)': [round(kurt_before[f], 4) for f in features_to_analyze]
})

print(stats_before_df.to_string(index=False))
print()
# 3. ЛОГАРИФМИЧЕСКОЕ ПРЕОБРАЗОВАНИЕ ДЛЯ ПРИЗНАКОВ С СИЛЬНОЙ АСИММЕТРИЕЙ

print("3. Логарифмическое преобразование для признаков с сильной асимметрией")
print("-" * 60)

# Определяем признаки с сильной асимметрией (|skew| > 1)
highly_skewed = [f for f in features_to_analyze if abs(skew_before[f]) > 1]
print(f"Признаки с сильной асимметрией (|skew| > 1): {highly_skewed}")
print()

if len(highly_skewed) > 0:
    # Применяем log1p(x) = log(1 + x) — безопасно даже если есть нули
    df_log = df.copy()
    for feat in highly_skewed:
        df_log[f'{feat}_log'] = np.log1p(df_log[feat])
    
    print(f"Созданы признаки: {[f'{f}_log' for f in highly_skewed]}")
    print()
    
    # Гистограммы ПОСЛЕ логарифмирования
    fig, axes = plt.subplots(1, len(highly_skewed), figsize=(6 * len(highly_skewed), 5))
    if len(highly_skewed) == 1:
        axes = [axes]
    
    for idx, feat in enumerate(highly_skewed):
        sns.histplot(
            df_log[f'{feat}_log'],
            kde=True,
            bins=30,
            color='darkorange',
            ax=axes[idx]
        )
        axes[idx].set_title(f'Распределение {feat} (log1p)', fontsize=12)
        axes[idx].set_xlabel(f'{feat}_log')
        axes[idx].set_ylabel('Частота')
    
    plt.tight_layout()
    plt.show()
    
    # Асимметрия и эксцесс ПОСЛЕ логарифмирования
    print("Асимметрия и эксцесс ПОСЛЕ логарифмирования:")
    print("-" * 60)
    
    skew_after = {}
    kurt_after = {}
    
    for feat in highly_skewed:
        skew_after[feat] = stats.skew(df_log[f'{feat}_log'])
        kurt_after[feat] = stats.kurtosis(df_log[f'{feat}_log'])
    
    stats_after_df = pd.DataFrame({
        'Признак': [f'{f}_log' for f in highly_skewed],
        'Skew (после)': [round(skew_after[f], 4) for f in highly_skewed],
        'Kurtosis (после)': [round(kurt_after[f], 4) for f in highly_skewed]
    })
    
    print(stats_after_df.to_string(index=False))
    print()
    
    # Сводная таблица ДО и ПОСЛЕ
    print("Сводная таблица: skew/kurtosis ДО и ПОСЛЕ логарифмирования")
    print("-" * 60)
    
    comparison_table = pd.DataFrame({
        'Признак': highly_skewed,
        'Skew ДО': [round(skew_before[f], 4) for f in highly_skewed],
        'Skew ПОСЛЕ': [round(skew_after[f], 4) for f in highly_skewed],
        'Kurtosis ДО': [round(kurt_before[f], 4) for f in highly_skewed],
        'Kurtosis ПОСЛЕ': [round(kurt_after[f], 4) for f in highly_skewed]
    })
    
    print(comparison_table.to_string(index=False))
else:
    print("Признаков с сильной асимметрией не обнаружено.")
    comparison_table = None

print()


# 4. СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ ПО ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (is_churn)

print("4. Наложение KDE для классов is_churn=0 и is_churn=1")
print("-" * 60)

# Выбираем 2-3 признака для сравнения
compare_features = ['age', 'income', 'years_experience']

fig, axes = plt.subplots(1, len(compare_features), figsize=(6 * len(compare_features), 5))
if len(compare_features) == 1:
    axes = [axes]

for idx, feat in enumerate(compare_features):
    # Класс 0 (не ушел)
    sns.kdeplot(
        df[df['is_churn'] == 0][feat],
        label='is_churn = 0 (не ушел)',
        color='blue',
        fill=True,
        alpha=0.4,
        ax=axes[idx]
    )
    # Класс 1 (ушел)
    sns.kdeplot(
        df[df['is_churn'] == 1][feat],
        label='is_churn = 1 (ушел)',
        color='red',
        fill=True,
        alpha=0.4,
        ax=axes[idx]
    )
    axes[idx].set_title(f'Распределение {feat} по классам', fontsize=12)
    axes[idx].set_xlabel(feat)
    axes[idx].set_ylabel('Плотность')
    axes[idx].legend()

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ВЫВОДЫ ПО АНАЛИЗУ РАСПРЕДЕЛЕНИЙ")
print("=" * 60)

print("""
1. Асимметрия признаков:
""")

for feat in features_to_analyze:
    skew_val = skew_before[feat]
    if abs(skew_val) > 1:
        direction = "правосторонняя" if skew_val > 0 else "левосторонняя"
        print(f"   • {feat}: сильная {direction} асимметрия (skew={skew_val:.4f})")
    elif abs(skew_val) > 0.5:
        direction = "правосторонняя" if skew_val > 0 else "левосторонняя"
        print(f"   • {feat}: умеренная {direction} асимметрия (skew={skew_val:.4f})")
    else:
        print(f"   • {feat}: распределение близко к симметричному (skew={skew_val:.4f})")





#1. Корелляция Пирсона между числовыми признаками и целевой переменной
print("Корреляция Пирсона между числовыми признаками и целевой перменной")
#Собираем все числовые признаки + целевую переменную в одну таблицу
df_for_corr = X.copy()
df_for_corr['is_churn'] = y

#Считаем матрицу корреляции Пирсона
corr_matrix = df_for_corr.corr(method='pearson')

#Берём только столбец с is_churn, это корреляция каждого признака с целью
corr_with_target = corr_matrix['is_churn'].drop('is_churn')

#Сортируем по абсолютному значению (от самой сильной связи к слабой)
corr_with_target_sorted = corr_with_target.abs().sort_values(ascending=False)
corr_with_target_sorted = corr_with_target[corr_with_target_sorted.index]
print("Корелляция признаков с целевой переменной")
print(corr_with_target_sorted.round(4))
print()

#2. Тепловая карта коррелляций

print("Тепловая карта коррелляций")
plt.figure(figsize=(12, 8))

# Рисуем heatmap всей матрицы корреляций
sns.heatmap(
    corr_matrix,
    annot=True,          # показывать числа в ячейках
    fmt='.2f',           # формат чисел (2 знака после запятой)
    cmap='coolwarm',     # цветовая схема: синий (отриц.) — белый — красный (полож.)
    center=0,            # центр colormap на 0
    vmin=-1, vmax=1,     # границы шкалы
    square=True,         # квадратные ячейки
    linewidths=0.5       # тонкие линии между ячейками
)

plt.title('Тепловая карта корреляций Пирсона', fontsize=14)
plt.tight_layout()
plt.show()

#3. Выделение признаков с высокой корреляцией

print("Признаки с высокой корреляцией с целевой переменной (r > 0.3)")

threshold = 0.3
high_corr_with_target = corr_with_target[corr_with_target.abs() > threshold]

if len(high_corr_with_target) > 0:
    print(f"Признаки с корреляцией > {threshold}:")
    for feat, val in high_corr_with_target.items():
        direction = 'полжительная' if val > 0 else 'отрицательная'
        print(f"   {feat:30s}  r = {val:+.4f}  ({direction})")
else:
    print(f" Ни один признак не имеет корреляции с целью выше {threshold}")
    print("  Это может означать, что признаки слабо информативны для предсказания churn,")
    print("  либо связь нелинейная (Пирсон её не ловит).")

print()

#Сильно коррелирующие друг с другом признаки
print("Признаки, сильно коррелирующие ДРУГ С ДРУГОМ (r > 0.5, без диагонали):")

#Создаём маску: убираем диагональ и нижний треугольник (чтобы не дублировать пары)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

#Ищем пары с |r|>0.5
high_corr_pairs = []
features = corr_matrix.columns
for i in range(len(features)):
    for j in range (i + 1, len(features)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.5:
            high_corr_pairs.append((features[i], features[j], r))
    
if len(high_corr_pairs) > 0:
    for f1, f2, r in high_corr_pairs:
        print(f" {f1:30s} ↔ {f2:30s}  r = {r:+.4f}")
print()
