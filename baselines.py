import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

# 1. Load the data we saved in Step 1
df = pd.read_csv("power_plant.csv")

# Which inputs relate most to power? (-1 to +1, further from 0 = stronger link)
print("Correlation of each column with PE:")
print(df.corr()["PE"].round(3))
print()

X = df[["AT", "V", "AP", "RH"]].values   # inputs
y = df["PE"].values                       # what we predict

# 2. Split: 80% for training, 20% hidden for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training rows:", len(X_train), "| Test rows:", len(X_test))
print()

# 3. Scale: learn the scaling from TRAINING rows only
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# 4. A helper that trains a model and prints its scores
def evaluate(name, model):
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = mean_absolute_percentage_error(y_test, pred) * 100
    r2 = r2_score(y_test, pred)
    print(f"{name}")
    print(f"  MAE : {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAPE: {mape:.3f} %")
    print(f"  R2  : {r2:.4f}")
    print()


# 5. Train and score the two baselines
evaluate("Linear Regression", LinearRegression())
evaluate("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))