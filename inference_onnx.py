import csv

import numpy as np
import onnxruntime as ort

# Читаем CSV стандартными средствами Python.
with open("data/dataset.csv", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    feature_names = [
        name for name in reader.fieldnames if name != "target"
    ]
    rows = list(reader)

# Берём по одному реальному образцу каждой категории.
examples = [
    next(row for row in rows if int(row["target"]) == category)
    for category in (0, 1, 2)
]

X = np.array(
    [
        [float(row[name]) for name in feature_names]
        for row in examples
    ],
    dtype=np.float32,
)
expected = np.array([int(row["target"]) for row in examples])

# Загружаем экспортированную модель.
session = ort.InferenceSession(
    "wine_model.onnx",
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
predicted, probabilities = session.run(None, {input_name: X})

print(f"Вход модели: {input_name}")
print(f"Размер входных данных: {X.shape}")
print(f"Правильные категории:    {expected}")
print(f"Предсказанные категории: {predicted}")
print("Вероятности категорий 0, 1, 2:")
print(np.round(probabilities, 4))
print(f"Совпадение ответов: {np.array_equal(expected, predicted)}")