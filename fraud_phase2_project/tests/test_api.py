"""Basic API contract tests."""
from fastapi.testclient import TestClient
from serving.app import app
from tests.tests_payload import SAMPLE_PAYLOAD

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()

def test_predict_endpoint():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["predicted_class"] in (0, 1)
    assert body["decision"] in ("fraud", "legitimate")

def test_missing_feature_rejected():
    payload = dict(SAMPLE_PAYLOAD)
    payload.pop("V1")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_extra_feature_rejected():
    payload = dict(SAMPLE_PAYLOAD)
    payload["unexpected"] = 123
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
