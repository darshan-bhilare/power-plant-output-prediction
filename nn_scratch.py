import numpy as np


class NeuralNet:
    """A simple fully-connected neural network, built with NumPy only.

    layer_sizes example: [4, 32, 16, 1] means
      4 inputs -> 32 neurons -> 16 neurons -> 1 output
    activation: 'tanh' or 'relu'
    """

    def __init__(self, layer_sizes, seed=0, activation="tanh"):
        self.activation = activation
        rng = np.random.default_rng(seed)
        self.W = []   # weights, one matrix per layer
        self.b = []   # biases, one row per layer
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
            self.W.append(rng.normal(0, scale, size=(n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))

    def _act(self, z):
        if self.activation == "relu":
            return np.maximum(0, z)
        return np.tanh(z)

    def _act_grad(self, a):
        """Derivative of the activation, written in terms of its output a."""
        if self.activation == "relu":
            return (a > 0).astype(a.dtype)
        return 1 - a ** 2

    def forward(self, X):
        """Push inputs through the network and return predictions."""
        self.A = [X]            # remember each layer's output (needed for backprop)
        a = X
        last = len(self.W) - 1
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            a = z if i == last else self._act(z)   # no activation on the last layer
            self.A.append(a)
        return a

    def loss(self, X, y):
        """Mean squared error."""
        pred = self.forward(X)
        return np.mean((pred - y) ** 2)

    def backward(self, y):
        """Backpropagation. Call forward() first. Returns gradients."""
        n = y.shape[0]
        pred = self.A[-1]
        dz = 2 * (pred - y) / n

        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            grads_W[i] = self.A[i].T @ dz
            grads_b[i] = dz.sum(axis=0, keepdims=True)
            if i > 0:
                da = dz @ self.W[i].T
                dz = da * self._act_grad(self.A[i])
        return grads_W, grads_b

    def step(self, grads_W, grads_b, lr):
        """Plain gradient descent step."""
        for i in range(len(self.W)):
            self.W[i] -= lr * grads_W[i]
            self.b[i] -= lr * grads_b[i]

    def predict(self, X):
        return self.forward(X)


def gradient_check(net, X, y, eps=1e-5):
    """Compare backprop gradients with slow 'nudge it and see' gradients.
    Returns the worst relative difference found (should be tiny)."""
    net.forward(X)
    gW, gb = net.backward(y)
    worst = 0.0
    for params, grads in ((net.W, gW), (net.b, gb)):
        for p, g in zip(params, grads):
            for idx in np.ndindex(p.shape):
                old = p[idx]
                p[idx] = old + eps
                loss_plus = net.loss(X, y)
                p[idx] = old - eps
                loss_minus = net.loss(X, y)
                p[idx] = old
                numeric = (loss_plus - loss_minus) / (2 * eps)
                diff = abs(numeric - g[idx]) / max(1e-8, abs(numeric) + abs(g[idx]))
                worst = max(worst, diff)
    return worst


def train(net, X, y, epochs=100, lr=0.01, optimizer="adam",
          batch_size=32, l2=0.0, momentum=0.9, seed=0):
    """Train with mini-batches. optimizer: 'sgd', 'momentum' or 'adam'.
    Returns the full-data loss after each epoch."""
    rng = np.random.default_rng(seed)
    params = net.W + net.b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = X.shape[0]
    history = []

    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            net.forward(X[idx])
            gW, gb = net.backward(y[idx])
            gW = [g + l2 * w for g, w in zip(gW, net.W)]
            grads = gW + gb
            t += 1

            for k, (p, g) in enumerate(zip(params, grads)):
                if optimizer == "sgd":
                    p -= lr * g
                elif optimizer == "momentum":
                    m[k] = momentum * m[k] + g
                    p -= lr * m[k]
                elif optimizer == "adam":
                    m[k] = beta1 * m[k] + (1 - beta1) * g
                    v[k] = beta2 * v[k] + (1 - beta2) * g * g
                    m_hat = m[k] / (1 - beta1 ** t)
                    v_hat = v[k] / (1 - beta2 ** t)
                    p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        history.append(net.loss(X, y))
    return history