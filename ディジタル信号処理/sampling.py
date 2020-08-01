# %%
# 標本化 高速フーリエ変換
import numpy as np
import matplotlib.pyplot as plt
from math import floor


def signal(N, freq, sn=0):  # 元の信号 sin(ωt)
    n = np.arange(N)
    t = n / N
    noise = sn * np.random.randn(N)
    omega = 2 * np.pi * freq  # 角周波数
    return np.sin(omega * t) + noise


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
# 標本化定理
# 「 データ数(サンプリング周期) >= 元信号の周期 * 2 」 でサンプリングした時、元信号に復元できる。
for i in [128, 32, 18, 17, 16, 15, 14, 12, 8]:
    N = i  # データ数
    sample_wave = signal(N, 8, 0)  # サンプリング信号

    # 高速フーリエ変換(FFT)
    F = np.fft.fft(sample_wave)
    # F[floor(0.5 * N) + 1:] = 0 + 0j  # アンチエリアジング
    F_abs = np.abs(F)

    # # 高速逆フーリエ変換
    # G = F.copy()
    # g = np.fft.ifft(G)
    # g = g.real

    print("freqency: 8[Hz]")
    print("Number of samples: {}".format(N))

    # グラフ表示
    plt.figure(figsize=(8, 4))
    plt.xlabel('n')
    plt.ylabel('Signal')
    plt.title('sampling wave')
    plt.plot(sample_wave)
    plt.show()

    plt.title("frequency spectrum")
    plt.plot(F_abs)
    plt.show()

    # plt.title("Restored signal")
    # plt.plot(g)
    # plt.show()

    print("")

# %%
