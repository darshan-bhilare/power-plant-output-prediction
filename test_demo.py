import os
import sqlite3
import tempfile

# Use a throwaway database so tests never touch the real log
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_demo.db")

from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402

client = TestClient(main.app)
GOOD = {"AT": 14.96, "V": 41.76, "AP": 1024.07, "RH": 73.17}


def count_logged():
    with sqlite3.connect(main.DB_PATH) as conn:
        return conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]


def test_home_page_loads():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Power plant output prediction" in r.text


def test_ranges_and_examples():
    data = client.get("/ranges").json()
    for col in ["AT", "V", "AP", "RH"]:
        assert data["ranges"][col]["min"] < data["ranges"][col]["max"]
    assert len(data["examples"]) == 3


def test_preview_is_not_logged():
    before = count_logged()
    r = client.post("/predict?log=false", json=GOOD)
    assert r.status_code == 200
    assert count_logged() == before


def test_drift_endpoint_reports_a_status():
    r = client.get("/drift")
    assert r.status_code == 200
    assert r.json()["status"] in {"not_enough_data", "ok", "drift_detected"}


def test_drift_endpoint_flags_hotter_weather():
    rows = main.train_features.sample(60, random_state=1)
    for _, row in rows.iterrows():
        body = {c: float(row[c]) for c in ["AT", "V", "AP", "RH"]}
        body["AT"] += 10
        client.post("/predict", json=body)
    report = client.get("/drift").json()
    assert report["status"] == "drift_detected"
    assert report["columns"]["AT"]["drift"] is True
