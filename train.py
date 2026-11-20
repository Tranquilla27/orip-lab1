import matplotlib.pyplot as plt
import mlflow.sklearn
from mlflow.models import infer_signature
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split

# Загружаем именно наш CSV, версионируемый через DVC.
dataset = pd.read_csv("data/dataset.csv")

# X — характеристики вина, y — правильные категории.
X = dataset.drop(columns=["target"])
y = dataset["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Всего объектов: {len(dataset)}")
print(f"Признаков: {X.shape[1]}")
print(f"Для обучения: {len(X_train)}")
print(f"Для проверки: {len(X_test)}")

# Настройки модели.
# Адрес уже запущенного сервера.
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Общая папка-журнал для наших экспериментов с Wine.
mlflow.set_experiment("wine_classification_experiment")

n_estimators = 150
max_depth = 15

# Один Run — одно обучение с конкретными настройками.
with mlflow.start_run(
    run_name=f"forest_{n_estimators}_depth_{max_depth}"
):
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Описываем, какие данные модель принимает и что возвращает.
    signature = infer_signature(X_train, model.predict(X_train))

    # Сохраняем обученную модель и сведения о её окружении.
    mlflow.sklearn.log_model(
        sk_model=model,
        name="random_forest_model",
        signature=signature,
        input_example=X_train.head(2),
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

        # Сортируем признаки по важности.
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
    ).sort_values()

    # Строим горизонтальную столбчатую диаграмму.
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot.barh(ax=ax)

    ax.set_title("Wine: feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()

    # Отправляем изображение в артефакты текущего запуска.
    mlflow.log_figure(fig, "plots/feature_importance.png")
    plt.close(fig)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 macro: {f1:.4f}")
    print("Настройки и метрики сохранены в MLflow.")