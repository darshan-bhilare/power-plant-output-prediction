import os
import sqlite3
import tempfile

# Use a throwaway database so tests never touch the real log
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_requests.db")

from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402

client = TestClient(main.app)
GOOD = {"AT": 14.96, "V": 41.76, "AP": 1024.07, "RH": 73.17}


def count_logged():
    with sqlite3.connect(main.DB_PATH) as conn:
        return conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_gives_sensible_value():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    value = r.json()["predicted_power_MW"]
    assert 420 <= value <= 500   # training data ranges from about 420 to 496


def test_impossible_humidity_is_rejected():
    r = client.post("/predict", json={**GOOD, "RH": 500})
    assert r.status_code == 422


def test_missing_field_is_rejected():
    r = client.post("/predict", json={"AT": 14.96})
    assert r.status_code == 422


def test_request_is_logged():
    before = count_logged()
    client.post("/predict", json=GOOD)
    assert count_logged() == before + 1