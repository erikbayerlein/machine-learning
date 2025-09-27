import numpy as np

def ols(
    x: np.ndarray,
    y: np.ndarray
) -> np.ndarray :
    x_trans = x.T
    return np.invert(x @ x_trans) @ x_trans @ y
