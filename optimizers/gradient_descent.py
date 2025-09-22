import numpy as np
from typing import Dict

def gradient_descent(
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
    loss_hist = []

    for _ in range(iterations):
        y_hat = w0 + w1 * x
        e = y - y_hat

        w0 += alpha * (e.sum() / N)
        w1 += alpha * ((e * x).sum() / N)

        loss_hist.append(np.mean(e**2))

    return {"w0": w0, "w1": w1, "loss": np.array(loss_hist)}
