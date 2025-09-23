import numpy as np
from typing import Dict

def stochastic_descent(
    alpha: float,
    iterations: int,
    x: np.ndarray,
    y: np.ndarray,
    w0: float = 0.0,
    w1: float = 0.0,
) -> Dict:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    assert x.shape == y.shape, "x and y need to have the same size"

    N = x.size
    rng = np.random.default_rng()

    loss_hist = []

    for _ in range(iterations):
        idx = rng.permutation(N)
        for i in idx:
            x_i: float = float(x[i])
            y_hat_i: float = w0 + w1 * x_i
            e_i: float = float(y[i]) - y_hat_i

            w0 += alpha * e_i
            w1 += alpha * e_i * x_i

        y_hat = w0 + w1 * x
        loss_hist.append(np.mean((y - y_hat) ** 2))

    return {"w0": w0, "w1": w1, "loss": np.array(loss_hist)}
