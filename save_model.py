import os
import joblib
import numpy as np
import pandas as pd
from preprocess import load_training_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score,
)

FEATURES = ["AT", "V", "AP", "RH"]

# Same data and same split as every earlier step
X, y = load_training_data("power_plant.csv")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale, then predict, bundled as one object
model = Pipeline([
    ("scaler", StandardScaler()),
    ("forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
])
model.fit(X_train, y_train)

# Check the scores match Step 2
pred = model.predict(X_test)
print("MAE :", round(mean_absolute_error(y_test, pred), 3))
print("RMSE:", round(float(np.sqrt(mean_squared_error(y_test, pred))), 3))
print("MAPE:", round(mean_absolute_percentage_error(y_test, pred) * 100, 3), "%")
print("R2  :", round(r2_score(y_test, pred), 4))

# Save the pipeline, and the training inputs (for the drift check later)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
X_train.to_csv("models/train_features.csv", index=False)
print("\nSaved models/model.joblib and models/train_features.csv")

# Prove the saved file works: load it fresh and predict one row
loaded = joblib.load("models/model.joblib")
sample = pd.DataFrame([{"AT": 14.96, "V": 41.76, "AP": 1024.07, "RH": 73.17}])
print("Prediction for the first row of the data (true value is 463.26):",
      round(float(loaded.predict(sample)[0]), 2))