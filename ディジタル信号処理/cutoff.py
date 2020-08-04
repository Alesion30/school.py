# %%
# 高周波成分除去 カットオフ周波数
import numpy as np
import matplotlib.pyplot as plt
from math import floor

# 表示用
import sympy
from IPython.display import display
from IPython.display import Math


# %%
# データのパラメータ
N = 40000  # サンプル数
dt = 1 * 1e-5  # サンプリング間隔
fs = 1 / dt  # サンプリング周期

# 軸
t = np.arange(0, N*dt, dt)  # 時間軸
freq = np.linspace(0, 1.0/dt, N)  # 周波数軸

# 時間信号 パラメータ  ※ 配列の長さは全て統一する必要有
a = 100  # 直流成分
ampli = [1, 1]  # 振幅
freqency = [25, 50]  # 周波数
phirad = [0, 0.5]  # 位相

# 時間信号 生成
noise = 0.5 * np.random.randn(N)  # ノイズ
f = a
for i in range(len(ampli)):
    f += ampli[i]*np.cos(2*np.pi*freqency[i]*t+phirad[i])
f += noise

# 時間信号 数式表示
sympy.init_printing()
sym_t = sympy.Symbol('t')
sym_pi = sympy.Symbol('2π')
sym_f = a
for i in range(len(ampli)):
    sym_f += sympy.sin(sym_pi * freqency[i] * sym_t + np.round(phirad[i], 2))
display(sym_f)


# %%
# 高速フーリエ変換
F = np.fft.fft(f)

# 元波形をコピーする
G = F.copy()

# ローパス
fc = 200  # カットオフ周波数
G[((freq > fc))] = 0 + 0j

# 高速逆フーリエ変換
g = np.fft.ifft(G)

# 実部の値のみ取り出し
g = g.real


# %%
# 可視化
print("サンプリング周期: {}[Hz]".format(int(np.round(fs))))
print("信号最大周波数: {}[Hz]".format(max(freqency)))

# 元信号f ノイズなし
plt.title("Noise free signal")
plt.xlabel('t[s]')
plt.ylabel('amplitude')
plt.plot(t, f-noise)
plt.show()

# 元信号f ノイスあり
plt.title("Noisy signal")
plt.xlabel('t[s]')
plt.ylabel('amplitude')
plt.plot(t, f)
plt.show()

# 周波数スペクトルF
plt.title("frequency spectrum")
plt.xlabel('frequency[Hz]')
plt.ylabel('strength')
plt.plot(freq, F)
plt.show()

# 周波数スペクトルG
plt.title('Removes high-frequency')
plt.xlabel('frequency[Hz]')
plt.ylabel('strength')
plt.plot(freq, G)
plt.show()

# 元信号g
plt.title('Restored Signal')
plt.xlabel('t[s]')
plt.ylabel('amplitude')
plt.plot(t, g)
plt.show()


# %%
