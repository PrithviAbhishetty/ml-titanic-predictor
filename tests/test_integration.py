import pytest
import httpx
import subprocess
import sys
import socket
import time

BASE_URL = "http://127.0.0.1:8001"

valid_passenger = {
    "pclass": 1,
    "sex": "female",
    "age": 30,
    "sibsp": 0,
    "parch": 0,
    "fare": 100.0,
    "embarked": "S"
}

@pytest.fixture(scope="module")
def server():
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8001"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    for _ in range(60):
        try:
            with socket.create_connection(("127.0.0.1", 8001), timeout=1):
                break
        except OSError:
            time.sleep(1)
    else:
        process.terminate()
        raise RuntimeError("Server failed to start")

    yield

    process.terminate()
    process.wait(timeout=5)

def test_server_starts(server):
    response = httpx.get(f"{BASE_URL}/health")
    assert response.status_code == 200

def test_cors_vercel_origin(server):
    response = httpx.post(
        f"{BASE_URL}/predict",
        json=valid_passenger,
        headers={"Origin": "https://ml-titanic-predictor.vercel.app"}
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers

def test_cors_localhost_origin(server):
    response = httpx.post(
        f"{BASE_URL}/predict",
        json=valid_passenger,
        headers={"Origin": "http://localhost:5173"}
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers

def test_cors_disallowed_origin(server):
    response = httpx.post(
        f"{BASE_URL}/predict",
        json=valid_passenger,
        headers={"Origin": "https://malicious-site.com"}
    )
    assert "access-control-allow-origin" not in response.headers

def test_cors_preflight(server):
    response = httpx.options(
        f"{BASE_URL}/predict",
        headers={
            "Origin": "https://ml-titanic-predictor.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
    )
    assert response.status_code == 200

def test_predict_response_time(server):
    start = time.time()
    response = httpx.post(f"{BASE_URL}/predict", json=valid_passenger)
    elapsed = time.time() - start
    assert response.status_code == 200
    assert elapsed < 1.0

def test_predict_real_request(server):
    response = httpx.post(f"{BASE_URL}/predict", json=valid_passenger)
    assert response.status_code == 200
    data = response.json()
    assert "survived" in data
    assert "survival_probability" in data

def test_content_type_header(server):
    response = httpx.post(f"{BASE_URL}/predict", json=valid_passenger)
    assert "application/json" in response.headers["content-type"]

def test_health_content_type(server):
    response = httpx.get(f"{BASE_URL}/health")
    assert "application/json" in response.headers["content-type"]
