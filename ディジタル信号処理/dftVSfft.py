# %%
# 標本化 高速フーリエ変換
import numpy as np
from numpy.fft import fft as FFT
import matplotlib.pyplot as plt
import time


def signal(N, freq, sn=0):  # 元の信号 sin(ωt)
    n = np.arange(N)
    t = n / N
    noise = sn * np.random.randn(N)
    omega = 2 * np.pi * freq  # 角周波数
    return np.sin(omega * t) + noise


def DFT(data):  # 離散フーリエ変換
    res = []
    N = len(data)
    for k in range(N):
        w = np.exp(-1j * 2 * np.pi * k / float(N))
        X_k = 0
        for n in range(N):
            X_k += data[n] * (w ** n)
        res.append(abs(X_k))
    return np.array(res)


# %%
# 元信号
print("signal")
signal_wave = signal(2**10, 8, 0)
plt.figure(figsize=(8, 4))
plt.xlabel('n')
plt.ylabel('Signal')
plt.title('signal wave')
plt.plot(signal_wave)
plt.show()
print("")

# %%
freq = 2**3
N = 2**10  # データ数
sample_wave = signal(N, freq, 0)  # サンプリング信号

print("freqency: {}[Hz]".format(freq))
print("Number of samples: {}個".format(N))

# 離散フーリエ変換(DFT)
start = time.time()
D = DFT(sample_wave)
D[int(0.5 * N) + 1:] = 0 + 0j  # アンチエリアジング
dft_time = time.time() - start
print("dft time: {}[s]".format(dft_time))

# 高速フーリエ変換(FFT)
start = time.time()
F = FFT(sample_wave)
F[int(0.5 * N) + 1:] = 0 + 0j  # アンチエリアジング
F_abs = np.abs(F)
fft_time = time.time() - start
print("fft time: {}[s]".format(fft_time))

ratio = dft_time / fft_time
print("ratio: {}倍".format(int(ratio)))

# %%
print("freqency: {}[Hz]".format(freq))
print("Number of samples: {}個".format(N))

# グラフ表示
plt.figure(figsize=(8, 4))
plt.xlabel('n')
plt.ylabel('Signal')
plt.title('sampling wave')
plt.plot(sample_wave)
plt.show()

plt.title("frequency spectrum (DFT)")
plt.plot(D)
plt.show()
print("dft time: {}[s]".format(dft_time))

print("")

plt.title("frequency spectrum (FFT)")
plt.plot(F_abs)
plt.show()
print("fft time: {}[s]".format(fft_time))


# %%
n = np.arange(N)  # データ数

# n^2
y_dft = n ** 2

# (n/2)*Log2(N)
y_fft = n / 2 * np.log2(n)

plt.xlabel('N')
plt.ylabel('times')
plt.plot(y_dft)
plt.plot(y_fft)
plt.show()

# 比率 dft/fft
ratio = dft_time / fft_time

print("real: {}".format(ratio))

y_dft = N ** 2
y_fft = N / 2 * np.log2(N)

ratio = y_dft / y_fft
print("ideal: {}".format(ratio))

# %%
