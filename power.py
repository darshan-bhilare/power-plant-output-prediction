import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score,
)
from nn_scratch import NeuralNet, train

# 1. Same data and same split as the baselines
df = pd.read_csv("power_plant.csv")
X = df[["AT", "V", "AP", "RH"]].values
y = df["PE"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Scale inputs AND target, learning the scaling from training rows only
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
X_train_s = x_scaler.transform(X_train)
X_test_s = x_scaler.transform(X_test)
y_train_s = y_scaler.transform(y_train)

# 3. Build and train the network
net = NeuralNet([4, 32, 16, 1], seed=0)
history = train(net, X_train_s, y_train_s, epochs=100, lr=0.003,
                optimizer="adam", batch_size=64)

for e in [0, 9, 24, 49, 74, 99]:
    print(f"epoch {e + 1:3d}   training loss (scaled) = {history[e]:.5f}")
print()

# 4. Predict on the hidden test rows, convert back to real power units
pred_s = net.predict(X_test_s)
pred = y_scaler.inverse_transform(pred_s)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mape = mean_absolute_percentage_error(y_test, pred) * 100
r2 = r2_score(y_test, pred)

print("Our network (from scratch, Adam)")
print(f"  MAE : {mae:.3f}")
print(f"  RMSE: {rmse:.3f}")
print(f"  MAPE: {mape:.3f} %")
print(f"  R2  : {r2:.4f}")