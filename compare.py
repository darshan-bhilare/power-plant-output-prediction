import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score,
)
from nn_scratch import NeuralNet, train

# Same data, same split, same scaling as before
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

# The shared settings, so the comparison is fair
EPOCHS, LR, BATCH = 500, 0.003, 128

rows = []


def record(name, pred_s, seconds):
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1))
    rows.append({
        "model": name,
        "MAE": round(mean_absolute_error(y_test, pred), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
        "MAPE %": round(mean_absolute_percentage_error(y_test, pred) * 100, 3),
        "R2": round(r2_score(y_test, pred), 4),
        "seconds": round(seconds),
    })
    print(rows[-1])


# 1. Our own network
start = time.time()
net = NeuralNet([4, 32, 16, 1], seed=0, activation="tanh")
train(net, X_train_s, y_train_s, epochs=EPOCHS, lr=LR,
      optimizer="adam", batch_size=BATCH)
record("From scratch (NumPy)", net.predict(X_test_s), time.time() - start)

# 2. PyTorch
start = time.time()
torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(4, 32), nn.Tanh(),
    nn.Linear(32, 16), nn.Tanh(),
    nn.Linear(16, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
Xt = torch.tensor(X_train_s, dtype=torch.float32)
yt = torch.tensor(y_train_s, dtype=torch.float32)
n = len(Xt)
for epoch in range(EPOCHS):
    perm = torch.randperm(n)
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        optimizer.zero_grad()
        loss = loss_fn(model(Xt[idx]), yt[idx])
        loss.backward()
        optimizer.step()
model.eval()
with torch.no_grad():
    pred_s = model(torch.tensor(X_test_s, dtype=torch.float32)).numpy()
record("PyTorch", pred_s, time.time() - start)

# 3. scikit-learn
start = time.time()
mlp = MLPRegressor(
    hidden_layer_sizes=(32, 16), activation="tanh", solver="adam",
    learning_rate_init=LR, batch_size=BATCH, max_iter=EPOCHS,
    alpha=0.0, n_iter_no_change=EPOCHS, random_state=0,
)
mlp.fit(X_train_s, y_train_s.ravel())
record("scikit-learn MLP", mlp.predict(X_test_s), time.time() - start)

# 4. The baselines from Step 2, for reference
print()
results = pd.DataFrame(rows)
results.to_csv("results_compare.csv", index=False)
print(results.to_string(index=False))
print()
print("Baselines from Step 2:  Linear Regression R2 0.9301 | Random Forest R2 0.9637")