import sqlite3
from datetime import datetime, timezone

import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

DB_PATH = os.environ.get("DB_PATH", "requests.db")
FEATURES = ["AT", "V", "AP", "RH"]

# Load the saved pipeline once, when the service starts
model = joblib.load("models/model.joblib")

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
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS requests (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   timestamp TEXT,
                   AT REAL, V REAL, AP REAL, RH REAL,
                   prediction REAL
               )"""
        )


init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(item: PlantInput):
    row = pd.DataFrame([item.model_dump()], columns=FEATURES)
    prediction = float(model.predict(row)[0])

    # Log every request so we can check for drift later
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO requests (timestamp, AT, V, AP, RH, prediction) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(),
             item.AT, item.V, item.AP, item.RH, prediction),
        )

    return {"predicted_power_MW": round(prediction, 2)}