from pathlib import Path

import mlflow
import mlflow.sklearn
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Загружаем конкретную зарегистрированную версию.
model_uri = "models:/wine_production_model/1"
print(f"Загрузка модели: {model_uri}")
model = mlflow.sklearn.load_model(model_uri)

# Любое число объектов, по 13 числовых признаков у каждого.
input_type = [("float_input", FloatTensorType([None, 13]))]

onnx_model = convert_sklearn(
    model,
    initial_types=input_type,
    target_opset=15,
    options={id(model): {"zipmap": False}},
)

# Проверяем корректность структуры ONNX.
onnx.checker.check_model(onnx_model)

output_path = Path("wine_model.onnx")
output_path.write_bytes(onnx_model.SerializeToString())

print(f"Модель сохранена: {output_path}")
print("Структура ONNX проверена.")