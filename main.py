import numpy as np
from optimizers.gradient_descent import gradient_descent

def main():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 100)
    y = 3.5 + 2.0 * x + rng.normal(0, 1, size=x.size)

    res = gradient_descent(alpha=0.01, iterations=1000, x=x, y=y, w0=0.0, w1=0.0)
    print(res["w0"], res["w1"])
    print(res["loss"][-1])


if __name__ == "__main__":
    main()
