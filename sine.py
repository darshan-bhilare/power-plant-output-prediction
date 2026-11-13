import numpy as np
import matplotlib.pyplot as plt
from nn_scratch import NeuralNet, gradient_check

# ---- Part 1: gradient check on a tiny network ----
rng = np.random.default_rng(1)
X_small = rng.normal(size=(10, 1))
y_small = np.sin(X_small)

check_net = NeuralNet([1, 5, 1], seed=0)
worst = gradient_check(check_net, X_small, y_small)
print(f"Gradient check, worst difference: {worst:.2e}")
print("PASS" if worst < 1e-6 else "FAIL - backprop has a bug")
print()

# ---- Part 2: learn the sine curve, trying different step sizes ----
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = np.sin(X)

best_net, best_loss, best_lr = None, float("inf"), None

for lr in [0.1, 0.03, 0.01]:
    net = NeuralNet([1, 16, 1], seed=0)
    for epoch in range(10001):
        net.forward(X)
        gW, gb = net.backward(y)
        net.step(gW, gb, lr=lr)
        if epoch % 2000 == 0:
            print(f"lr={lr}  epoch {epoch:5d}   loss = {net.loss(X, y):.6f}")
    final = net.loss(X, y)
    print(f"--> lr={lr} final loss: {final:.6f}\n")
    if final < best_loss:
        best_net, best_loss, best_lr = net, final, lr

print(f"Best learning rate: {best_lr}")

# ---- Part 3: draw the best result ----
plt.plot(X, y, label="True sine curve")
plt.plot(X, best_net.predict(X), "--", label=f"Our network (lr={best_lr})")
plt.legend()
plt.title("Neural network from scratch learning sin(x)")
plt.savefig("sine_fit.png")
plt.show()