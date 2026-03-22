from fastapi.testclient import TestClient
import os
import sys

from api.main import app

client = TestClient(app)

# Valid passenger payload for re-use across tests
valid_passenger = {
    "pclass": 1,
    "sex": "female",
    "age": 30,
    "sibsp": 0,
    "parch": 0,
    "fare": 100.0,
    "embarked": "S"
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_health_head():
    response = client.head("/health")
    assert response.status_code == 200

def test_predict_returns_200():
    response = client.post("/predict", json=valid_passenger)
    assert response.status_code == 200

def test_predict_response_structure():
    response = client.post("/predict", json=valid_passenger)
    data = response.json()
    assert "survived" in data
    assert "survival_probability" in data
    assert isinstance(data["survived"], bool)
    assert isinstance(data["survival_probability"], float)
    
def test_predict_invalid_probability_range():
    response = client.post("/predict", json=valid_passenger)
    data = response.json()
    assert 0.0 <= data["survival_probability"] <= 1.0

def test_predict_invalid_pclass():
    passenger = valid_passenger.copy()
    passenger["pclass"] = 5
    response = client.post("/predict", json=passenger)
    assert response.status_code == 422

def test_predict_invalid_sex():
    passenger = valid_passenger.copy()
    passenger["sex"] = "unknown"
    response = client.post("/predict", json=passenger)
    assert response.status_code == 422

def test_predict_invalid_embarked():
    passenger = valid_passenger.copy()
    passenger["embarked"] = "Z"
    response = client.post("/predict", json=passenger)
    assert response.status_code == 422

def test_predict_missing_field():
    passenger = valid_passenger.copy()
    del passenger["age"]
    response = client.post("/predict", json=passenger)
    assert response.status_code == 422

def test_predict_negative_age():
    passenger = valid_passenger.copy()
    passenger["age"] = -5.0
    response = client.post("/predict", json=passenger)
    assert response.status_code == 422

def test_predict_age_as_string():
    passenger = valid_passenger.copy()
    passenger["age"] = "twenty nine"
    response = client.post("/predict", json=passenger)
    assert response.status_code == 422

def test_predict_negative_fare():
    passenger = valid_passenger.copy()
    passenger["fare"] = -50.0
    response = client.post("/predict", json=passenger)
    assert response.status_code == 422

def test_predict_empty_body():
    response = client.post("/predict", json={})
    assert response.status_code == 422