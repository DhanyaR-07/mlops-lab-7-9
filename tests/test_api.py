import os
import pytest
from fastapi.testclient import TestClient
from src.api import app

pytestmark = pytest.mark.skipif(not os.path.exists("models/model.joblib"),
                                reason="Run python -m src.train first")

VALID = {"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.98, "AveBedrms": 1.02,
         "Population": 322.0, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23}


def test_health_and_valid_prediction():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict", json=VALID)
        assert r.status_code == 200
        assert len(r.json()["prediction"]) == 1


def test_invalid_requests():
    with TestClient(app) as c:
        missing = {k: v for k, v in VALID.items() if k != "MedInc"}
        assert c.post("/predict", json=missing).status_code == 422
        assert c.post("/predict", json={**VALID, "HouseAge": "abc"}).status_code == 422
        assert c.post("/predict", json={**VALID, "MedInc": -5}).status_code == 422