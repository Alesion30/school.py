# 座標系(x, y)を極座標系(r, θ)に変換
import numpy as np


def ph(x, y):
    if x == 0:
        angle = 90 * y / np.abs(y)
        r = np.abs(y)
    else:
        theta = np.arctan(y / x)
        angle = np.rad2deg(theta)
        r = np.sqrt(x**2 + y**2)

    return [np.round(r, 2), np.round(angle, 2)]


if __name__ == "__main__":
    # I0
    x = -0.67
    y = -0.33
    print(ph(x, y))

    # I1
    x = 1.91
    y = 1.82
    print(ph(x, y))

    # I2
    x = 0.76
    y = -0.49
    print(ph(x, y))
