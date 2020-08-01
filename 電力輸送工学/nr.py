# NR法 関数の解を求める
import numpy as np

def f(x):
    return x**2-2*np.sin(x)

def divf(x):
    return 2*x-2*np.cos(x)

# 初期値
x = [1]

# 相対誤差
z = ["-"]

# 許容誤差
ips = 10**(-4)

xi = x[0]
zi = ips + 1
i = 0
print("初期値: " + str(xi))
print("許容誤差: " + str(ips))
print("")
while zi > ips:
    print(str(i + 1) + "周目")
    xi = x[i] - f(x[i]) / divf(x[i])
    print("x: " + str(xi))
    x.append(xi)
    zi = abs((x[i + 1] - x[i]) / x[i])
    print("z: " + str(zi))
    z.append(zi)
    i += 1
    print("")

print("解: " + str(xi))
print("誤差: " + str(zi))
