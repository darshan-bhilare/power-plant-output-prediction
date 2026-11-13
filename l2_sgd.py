import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score,
)
from nn_scratch import NeuralNet, train

df = pd.read_csv("power_plant.csv")
X = df[["AT", "V", "AP", "RH"]].values
y = df["PE"].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
X_train_s = x_scaler.transform(X_train)
X_test_s = x_scaler.transform(X_test)
y_train_s = y_scaler.transform(y_train)

# name, optimizer, learning rate, L2 strength
experiments = [
    ("adam 0.003, L2 0",      "adam", 0.003, 0.0),
    ("adam 0.003, L2 0.0001", "adam", 0.003, 1e-4),
    ("adam 0.003, L2 0.001",  "adam", 0.003, 1e-3),
    ("adam 0.003, L2 0.01",   "adam", 0.003, 1e-2),
    ("sgd 0.03, L2 0",        "sgd",  0.03,  0.0),
    ("sgd 0.1, L2 0",         "sgd",  0.1,   0.0),
    ("momentum 0.003, L2 0",  "momentum", 0.003, 0.0),
]

rows = []
for name, opt, lr, l2 in experiments:
    start = time.time()
    net = NeuralNet([4, 32, 16, 1], seed=0, activation="tanh")
    train(net, X_train_s, y_train_s, epochs=500, lr=lr,
          optimizer=opt, batch_size=128, l2=l2)
    pred = y_scaler.inverse_transform(net.predict(X_test_s))
    pred_tr = y_scaler.inverse_transform(net.predict(X_train_s))
    rows.append({
        "experiment": name,
        "MAE": round(mean_absolute_error(y_test, pred), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
        "MAPE %": round(mean_absolute_percentage_error(y_test, pred) * 100, 3),
        "test R2": round(r2_score(y_test, pred), 4),
        "train R2": round(r2_score(y_train, pred_tr), 4),
        "seconds": round(time.time() - start),
    })
    print(rows[-1])

results = pd.DataFrame(rows)
results.to_csv("results_l2_sgd.csv", index=False)
print()
print(results.to_string(index=False))