from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import onnxruntime as ort
from fastapi import FastAPI, Response
import numpy as np


class WineRequestData(BaseModel):
    # Не принимаем неизвестные поля, NaN и бесконечность.
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    alcohol: float = Field(gt=0, examples=[14.23])
    malic_acid: float = Field(gt=0, examples=[1.71])
    ash: float = Field(gt=0, examples=[2.43])
    alcalinity_of_ash: float = Field(gt=0, examples=[15.6])
    magnesium: float = Field(gt=0, examples=[127.0])
    total_phenols: float = Field(gt=0, examples=[2.8])
    flavanoids: float = Field(gt=0, examples=[3.06])
    nonflavanoid_phenols: float = Field(gt=0, examples=[0.28])
    proanthocyanins: float = Field(gt=0, examples=[2.29])
    color_intensity: float = Field(gt=0, examples=[5.64])
    hue: float = Field(gt=0, examples=[1.04])
    od280_od315_of_diluted_wines: float = Field(
        gt=0,
        examples=[3.92],
    )
    proline: float = Field(gt=0, examples=[1065.0])


class WineResponseData(BaseModel):
    predicted_class: int = Field(ge=0, le=2)
    probabilities: list[float] = Field(
        min_length=3,
        max_length=3,
    )
    model_status: str


MODEL_PATH = Path(__file__).resolve().parent / "wine_model.onnx"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Выполняется при запуске сервера.
    print("Загрузка модели ONNX в память...")

    app.state.onnx_session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    print("Модель загружена, сервер готов к работе.")

    yield

    # Выполняется при остановке сервера.
    print("Освобождение сессии ONNX...")
    app.state.onnx_session = None


app = FastAPI(
    title="Wine Classification API",
    version="1.0",
    lifespan=lifespan,
)


MODEL_VERSION = "wine_onnx_v1.1_updated"

PREDICTIONS_COUNTER = Counter(
    "predictions_total",
    "Number of successful predictions",
    ["predicted_class", "model_version"],
)

INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Time spent preparing input and running inference",
    buckets=(
        0.001, 0.005, 0.01, 0.025,
        0.05, 0.1, 0.25, 0.5, 1.0,
    ),
)

@app.post("/v1/predict", response_model=WineResponseData)
def predict(data: WineRequestData):
    # Порядок признаков должен совпадать с порядком при обучении.
    start_time = perf_counter()
    input_array = np.array(
        [[
            data.alcohol,
            data.malic_acid,
            data.ash,
            data.alcalinity_of_ash,
            data.magnesium,
            data.total_phenols,
            data.flavanoids,
            data.nonflavanoid_phenols,
            data.proanthocyanins,
            data.color_intensity,
            data.hue,
            data.od280_od315_of_diluted_wines,
            data.proline,
        ]],
        dtype=np.float32,
    )

    session = app.state.onnx_session
    input_name = session.get_inputs()[0].name

    labels, probabilities = session.run(
        None,
        {input_name: input_array},
    )

    predicted_class = int(labels[0])

    duration = perf_counter() - start_time
    INFERENCE_LATENCY.observe(duration)

    PREDICTIONS_COUNTER.labels(
        predicted_class=str(predicted_class),
        model_version=MODEL_VERSION,
    ).inc()

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities[0].tolist(),
        "model_status": MODEL_VERSION,
    }

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )