# syntax=docker/dockerfile:1

# Этап 1: подготовка библиотек
FROM python:3.14-slim AS builder
WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN pip wheel --no-cache-dir \
    --wheel-dir=/build/wheels \
    -r requirements-api.txt

# Этап 2: среда запуска приложения
FROM python:3.14-slim AS production
WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY --from=builder /build/requirements-api.txt .

RUN --mount=type=bind,from=builder,source=/build/wheels,target=/wheels \
    pip install --no-cache-dir --no-index \
    --find-links=/wheels -r requirements-api.txt

COPY wine_model.onnx .
COPY main.py .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]