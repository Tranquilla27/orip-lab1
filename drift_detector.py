from pathlib import Path
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "dataset.csv"


def prepare_datasets():
    reference = pd.read_csv(DATA_PATH).drop(columns=["target"])

    # Значения идут в том же порядке, что и признаки исходного CSV.
    shifted_values = np.array([
        12.8, 5.0, 2.5, 25.0, 100,
        0.6, 0.1, 0.7, 0.3, 14.0,
        0.35, 1.0, 500,
    ])

    rng = np.random.default_rng(42)

    # Небольшие вариации вокруг смещённого образца.
    noise = rng.uniform(0.98, 1.02, size=(900, 13))
    shifted = pd.DataFrame(
        shifted_values * noise,
        columns=reference.columns,
    )

    normal = reference.sample(
        n=100,
        replace=True,
        random_state=42,
    )

    current = pd.concat(
        [normal, shifted],
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    return reference, current


if __name__ == "__main__":
    reference, current = prepare_datasets()

    print(f"Эталон: {len(reference)} строк")
    print(f"Текущая выборка: {len(current)} строк")
    print("Текущие данные: искусственная симуляция 90% смещённых образцов")

    definition = DataDefinition(
        numerical_columns=list(reference.columns),
    )

    report = Report([
        DataDriftPreset(
            method="ks",
            threshold=0.05,
            drift_share=0.5,
        )
    ])

    snapshot = report.run(
        reference_data=Dataset.from_pandas(
            reference,
            data_definition=definition,
        ),
        current_data=Dataset.from_pandas(
            current,
            data_definition=definition,
        ),
    )

    report_path = Path(__file__).resolve().parent / "data_drift_report.html"
    snapshot.save_html(str(report_path))

    metrics = snapshot.dict()["metrics"]
    summary = next(
        metric["value"]
        for metric in metrics
        if metric["config"]["type"].endswith(":DriftedColumnsCount")
    )

    count = int(summary["count"])
    share = summary["share"]

    print(f"Всего признаков: {len(reference.columns)}")
    print(f"Признаков с дрейфом: {count}")
    print(f"Доля признаков с дрейфом: {share:.2%}")

    if share >= 0.5:
        print("АЛЕРТ: обнаружен существенный дрейф входных данных!")
        print("Нужно проверить источник данных и качество модели.")
    else:
        print("Порог алерта по доле признаков не достигнут.")

    print(f"HTML-отчёт: {report_path}")