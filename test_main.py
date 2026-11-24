import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def payload():
    return {
        "alcohol": 14.23,
        "malic_acid": 1.71,
        "ash": 2.43,
        "alcalinity_of_ash": 15.6,
        "magnesium": 127,
        "total_phenols": 2.8,
        "flavanoids": 3.06,
        "nonflavanoid_phenols": 0.28,
        "proanthocyanins": 2.29,
        "color_intensity": 5.64,
        "hue": 1.04,
        "od280_od315_of_diluted_wines": 3.92,
        "proline": 1065,
    }

def test_predict_success(client, payload):
    response = client.post("/v1/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_class"] == 0
    assert result["model_status"] == "wine_onnx_v1.1_updated"

    probabilities = result["probabilities"]
    assert len(probabilities) == 3
    assert all(0 <= value <= 1 for value in probabilities)
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-5)
    assert probabilities[0] == max(probabilities)

def test_predict_negative_value(client, payload):
    payload["alcohol"] = -5.1

    response = client.post("/v1/predict", json=payload)

    assert response.status_code == 422

    errors = response.json()["detail"]
    assert any(
        error["loc"] == ["body", "alcohol"]
        and error["type"] == "greater_than"
        for error in errors
    )


def test_predict_invalid_type(client, payload):
    payload["magnesium"] = "много"

    response = client.post("/v1/predict", json=payload)

    assert response.status_code == 422

    errors = response.json()["detail"]
    assert any(
        error["loc"] == ["body", "magnesium"]
        and error["type"] == "float_parsing"
        for error in errors
    )