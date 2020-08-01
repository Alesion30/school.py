# 方形波
import matplotlib.pyplot as plt
import numpy as np


# 時間
t = np.linspace(0, 10, 100)

# 周波数[Hz]
f = 10 ** 3

# 角周波数
omega = 2 * np.pi * f

# 波
y = []
for i in range(0, 1000):
    if i % 2 == 1:
        y.append((1 / i) * np.sin(i * omega * t))
    else:
        y.append(0)

# plot
plt.plot(t, y[1], label="base")
plt.plot(t, y[1] + y[3], label="+3dim")
plt.plot(t, y[1] + y[3] + y[5], label="+5dim")
plt.plot(t, y[1] + y[3] + y[5] + y[7], label="+7dim")
plt.plot(t, y[1] + y[3] + y[5] + y[7] + y[9], label="+9dim")
plt.legend()
plt.show()


## 方形波
swave = sum(y)
plt.plot(t, swave)
plt.show()
