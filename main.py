import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from drift import check_drift, load_requests
from preprocess import FEATURES, preprocess_features

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.environ.get("DB_PATH", "requests.db")
MIN_LOGGED = 30      # the drift check needs at least this many logged requests
WINDOW = 500         # and looks at no more than the most recent this many

# Load the saved pipeline and the training inputs once, when the service starts
model = joblib.load(BASE_DIR / "models" / "model.joblib")
# One request is tiny, so spreading it over all CPU cores only adds overhead
model.set_params(forest__n_jobs=1)
train_features = pd.read_csv(BASE_DIR / "models" / "train_features.csv")

RANGES = {
    c: {"min": round(float(train_features[c].min()), 2),
        "max": round(float(train_features[c].max()), 2)}
    for c in FEATURES
}


def pick_examples(df):
    """Three real readings from the training data: cool, typical and hot."""
    ordered = df.sort_values("AT").reset_index(drop=True)
    examples = []
    for label, q in [("Cool day", 0.10), ("Typical day", 0.50), ("Hot day", 0.90)]:
        row = ordered.iloc[int(q * (len(ordered) - 1))]
        examples.append({"label": label,
                         **{c: round(float(row[c]), 2) for c in FEATURES}})
    return examples


EXAMPLES = pick_examples(train_features)

app = FastAPI(title="Power Plant Output Prediction")


class PlantInput(BaseModel):
    """What a caller must send. The limits only reject impossible values,
    and are deliberately wider than the training data so that unusual
    (drifted) inputs still get through and show up in the drift check."""
    AT: float = Field(..., ge=-50, le=60, description="Temperature (deg C)")
    V: float = Field(..., ge=0, le=150, description="Exhaust vacuum (cm Hg)")
    AP: float = Field(..., ge=800, le=1100, description="Air pressure (millibar)")
    RH: float = Field(..., ge=0, le=105, description="Relative humidity (%)")


def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS requests (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   timestamp TEXT,
                   AT REAL, V REAL, AP REAL, RH REAL,
                   prediction REAL
               )"""
        )
        conn.commit()


init_db()


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ranges")
def ranges():
    """What the model learned from, so the demo page can show it."""
    return {"ranges": RANGES, "examples": EXAMPLES,
            "training_rows": len(train_features)}


@app.post("/predict")
def predict(item: PlantInput, log: bool = True):
    try:
        row = preprocess_features(pd.DataFrame([item.model_dump()]))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    prediction = float(model.predict(row)[0])

    # Log every real request so we can check for drift later.
    # The demo page sends log=false for its live previews.
    if log:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.execute(
                "INSERT INTO requests (timestamp, AT, V, AP, RH, prediction) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(),
                 item.AT, item.V, item.AP, item.RH, prediction),
            )
            conn.commit()

    return {"predicted_power_MW": round(prediction, 2)}


@app.get("/drift")
def drift_report():
    new = load_requests(DB_PATH).tail(WINDOW)
    n = len(new)
    if n < MIN_LOGGED:
        return {"status": "not_enough_data", "logged_requests": n,
                "needed": MIN_LOGGED}
    report = check_drift(train_features, new, min_rows=MIN_LOGGED)
    flagged = [c for c, r in report.items() if r["drift"]]
    return {"status": "drift_detected" if flagged else "ok",
            "logged_requests": n, "columns": report}