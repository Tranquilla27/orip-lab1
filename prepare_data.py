import argparse
from pathlib import Path

from sklearn.datasets import load_wine

parser = argparse.ArgumentParser()
parser.add_argument("--version", choices=["1", "2"], default="1")
args = parser.parse_args()

wine = load_wine(as_frame=True)
full_dataset = wine.frame

new_rows = full_dataset.groupby("target", group_keys=False).tail(5)

if args.version == "1":
    dataset = full_dataset.drop(index=new_rows.index)
else:
    dataset = full_dataset

Path("data").mkdir(exist_ok=True)
dataset.to_csv("data/dataset.csv", index=False)

print(f"Версия {args.version}.0: {len(dataset)} объектов")
print(f"Количество признаков: {len(wine.feature_names)}")
print("Количество объектов каждого класса:")
print(dataset["target"].value_counts().sort_index())