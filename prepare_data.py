from pathlib import Path

from sklearn.datasets import load_wine

# Загружаем встроенный датасет в виде таблицы.
wine = load_wine(as_frame=True)
full_dataset = wine.frame

# Откладываем 15 объектов для следующей версии данных:
# по 5 из каждой категории, чтобы сохранить все три класса.
new_rows = (
    full_dataset.groupby("target", group_keys=False)
    .tail(5)
)
dataset_v1 = full_dataset.drop(index=new_rows.index)

Path("data").mkdir(exist_ok=True)

dataset_v1.to_csv("data/dataset.csv", index=False)

print(f"Полный датасет: {len(full_dataset)} объектов")
print(f"Версия 1.0: {len(dataset_v1)} объектов")
print(f"Количество признаков: {len(wine.feature_names)}")
print("Количество объектов каждого класса:")
print(dataset_v1["target"].value_counts().sort_index())