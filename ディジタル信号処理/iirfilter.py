# IIRフィルタ
# https://qiita.com/trami/items/9553342d970443f5e663
# %%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# パラメータ設定
fs = 20 * 10**3                  # サンプリング周波数[Hz]
fpass = 3.4 * 10**3              # 通過遮断周波数[Hz]
fstop = 4.6 * 10**3              # 阻止域遮断周波数[Hz]
gpass = 0.4                      # 通過域最大損失量[dB]
gstop = 30                       # 阻止域最小減衰量[dB]

# 正規化
fn = fs/2                        # ナイキスト周波数
wp = fpass/fn                    # 正規化通過遮断周波数(無次元)
ws = fstop/fn                    # 正規化阻止域遮断周波数(無次元)

# %%
# フィルタ実装
# 実際に使用するパラメータ
fs = 20 * 10**3                  # サンプリング周波数[Hz]
wp = fpass/fn                    # 正規化通過遮断周波数(無次元)
ws = fstop/fn                    # 正規化阻止域遮断周波数(無次元)
gpass = 0.4                      # 通過域最大損失量[dB]
gstop = 30                       # 阻止域最小減衰量[dB]

l1, l2, l3, l4 = 'butter', 'cheby1', 'cheby2', 'ellip'

# ButterWorth filter

n1, wn1 = signal.buttord(wp, ws, gpass, gstop, fs=fs)

b1, a1 = signal.iirfilter(n1, wn1, rp=gpass, rs=gstop, btype='low', ftype=l1)


f1, h1 = signal.freqz(b1, a1, fs=fs)

# Chebychev1 filter
# n2, wn2 = signal.cheb1ord(wp, ws, gpass, gstop, fs=fs)
# b2, a2 = signal.iirfilter(n2, wn2, rp=gpass, rs=gstop, btype='low', ftype=l2)
# f2, h2 = signal.freqz(b2, a2, fs=fs)

# Chebychev2 filter
# n3, wn3 = signal.cheb2ord(wp, ws, gpass, gstop, fs=fs)
# b3, a3 = signal.iirfilter(n3, wn3, rp=gpass, rs=gstop, btype='low', ftype=l3)
# f3, h3 = signal.freqz(b3, a3, fs=fs)

# elliptic filter
# n4, wn4 = signal.ellipord(wp, ws, gpass, gstop, fs=fs)
# b4, a4 = signal.iirfilter(n4, wn4, rp=gpass, rs=gstop, btype='low', ftype=l4)
# f4, h4 = signal.freqz(b4, a4, fs=fs)

# %%
# 可視化
fig, ax = plt.subplots()

ax.semilogx(f1, 20 * np.log10(abs(h1)), label=l1 + '(n=' + str(n1) + ')')
# ax.semilogx(f2, 20*np.log10(abs(h2)), label=l2+'(n='+str(n2)+')')
# ax.semilogx(f3, 20*np.log10(abs(h3)), label=l3+'(n='+str(n3)+')')
# ax.semilogx(f4, 20*np.log10(abs(h4)), label=l4+'(n='+str(n4)+')')

plt.grid(which="both")
ax.set_title('Digital low pass filter frequency response')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Amplitude [dB]')
ax.legend(loc=0)

plt.show()

# %%
