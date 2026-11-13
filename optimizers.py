import numpy as np
from nn_scratch import NeuralNet, train

X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = np.sin(X)

settings = [
    ("sgd",      0.03),
    ("momentum", 0.01),
    ("adam",     0.01),
    ("adam",     0.003),
]

for name, lr in settings:
    net = NeuralNet([1, 16, 1], seed=0)
    history = train(net, X, y, epochs=1000, lr=lr, optimizer=name, batch_size=32)
    print(f"{name:9s} lr={lr:<6}  loss after 100 epochs: {history[99]:.6f}"
          f"   after 1000 epochs: {history[-1]:.6f}")