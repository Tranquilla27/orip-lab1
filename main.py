from pydantic import BaseModel, ConfigDict, Field
from contextlib import asynccontextmanager
from pathlib import Path

import onnxruntime as ort
from fastapi import FastAPI
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


@app.post("/v1/predict", response_model=WineResponseData)
def predict(data: WineRequestData):
    # Порядок признаков должен совпадать с порядком при обучении.
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

    return {
        "predicted_class": int(labels[0]),
        "probabilities": probabilities[0].tolist(),
        "model_status": "wine_onnx_v1.1_updated",
    }