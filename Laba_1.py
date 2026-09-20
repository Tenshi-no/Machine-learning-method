import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. ГЕНЕРАЦИЯ ДАТАСЕТА (ваш код) ---
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

print("РЕАЛИЗАЦИЯ НОВОГО ПРИЗНАКА")

# Признак income_per_year_exp = income / years_experience (если years_experience > 0)
df['income_per_year_exp'] = np.where(df['years_experience'] > 0, df['income']/df['years_experience'], 0)# избегаем деления на 0
df['income_per_year_exp'] = df['income_per_year_exp'].round(2)
# Бинарный признак high_income = income > median(income)
income_median = df['income'].median()
df['high_income'] = (df['income'] > income_median).astype(int)
# Признак уровня образования
df['education_level'] = pd.cut(df["education_encoded"], bins=[-1, 1.5, 2.5 ,5], labels=['Низкий','Средний','Высокий'])
print("Строки с новыми признаками")

preview_cols = ['income', 'years_experience', 'income_per_year_exp', 'high_income', 'education', 'education_level']
print(df[preview_cols].head(5))
print("-" * 60)
