import pandas as pd

df = pd.read_csv("titanic.csv")

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

