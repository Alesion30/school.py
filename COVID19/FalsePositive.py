import numpy as np
import matplotlib.pyplot as plt


def acc(p, n, s=0.7, d=0.99):
    """
    Calculate the accuracy of the inspection

    Parameters
    ----------
    p : float
        positivity
    n : int
        number of tests
    s : float
        sensitivity
    d : int
        singularity

    Returns
    -------
    p_acc : float
        Moderately Positive
    n_acc : float
        Moderately Negative
    """

    # 陽性判定された人数
    n_p = round(n * p)

    # 陰性判定された人数
    n_n = n - n_p

    # 正しく陽性だと判定された人数
    tp = round((n_n - n * d) / ((1 - d) * (1 / s - 1) - 1))

    # 間違って陰性だと判定された人数
    fn = round((1 / s - 1) * tp)

    # 間違って陽性だと判定された人数
    fp = max(n_p - tp, 0)

    # 正しく陰性だと判定された人数
    tn = max(n_n - fn, 0)

    # 陽性的中度
    p_acc = round(tp / n_p * 100, 1)

    # 陰性的中度
    n_acc = round(tn / n_n * 100, 1)

    return p_acc, n_acc


def main():
    # PCR parameter
    s = 0.7  # 感度 sensitivity
    d = 0.99  # 特異度 singularity

    # Data for Tokyo on August 18
    # https://stopcovid19.metro.tokyo.lg.jp/cards/positive-rate/
    p = 0.055  # 陽性率 positivity
    n = 4036  # 検査数 number of tests

    print("-------------------------------")
    print("東京都[08/18]のデータ")
    print("-------------------------------")
    print("陽性率 {}%".format(p * 100))
    print("検査数 {}人".format(n))
    print("-------------------------------")
    p_acc, n_acc = acc(p, n, s, d)
    print("陽性的中率(予想) {}%".format(p_acc))
    print("陰性的中率(予想) {}%".format(n_acc))
    print("-------------------------------")

    # # 検査数-偽陽性率 グラフ
    # P = [0.01, 0.05, 0.1]  # 陽性率
    # N = np.arange(1e03, 1e05, 1e02)  # 検査数
    # P_err = []  # 偽陽性率
    # N_err = []  # 偽陰性率
    # for i, p in enumerate(P):
    #     P_err.append([])
    #     N_err.append([])
    #     for n in N:
    #         p_acc, n_acc = acc(p, n, s, d)
    #         P_err[i].append(round(100 - p_acc))
    #         N_err[i].append(round(100 - n_acc))

    # plt.title("False Positive")
    # plt.xlabel('n')
    # plt.ylabel('error[%]')
    # for i, p in enumerate(P):
    #     plt.plot(N, P_err[i], label="p={}".format(p))
    # plt.legend()
    # plt.show()

    # plt.title("False Negative")
    # plt.xlabel('n')
    # plt.ylabel('error[%]')
    # for i, p in enumerate(P):
    #     plt.plot(N, N_err[i], label="p={}".format(p))
    # plt.legend()
    # plt.show()

    # 陽性率-偽陽性率 グラフ
    P = np.arange(0.01, 1.0, 0.01)  # 陽性率
    N = [1000, 5000, 10000]  # 検査数
    P_err = []  # 偽陽性率
    N_err = []  # 偽陰性率
    for i, n in enumerate(N):
        P_err.append([])
        N_err.append([])
        for p in P:
            p_acc, n_acc = acc(p, n, s, d)
            P_err[i].append(round(100 - p_acc))
            N_err[i].append(round(100 - n_acc))

    plt.title("False Positive")
    plt.xlabel('positivity[%]')
    plt.ylabel('error[%]')
    for i, n in enumerate(N):
        plt.plot(P * 100, P_err[i], label="n={}".format(n))
    plt.legend()
    plt.show()

    plt.title("False Negative")
    plt.xlabel('positivity[%]')
    plt.ylabel('error[%]')
    for i, n in enumerate(N):
        plt.plot(P * 100, N_err[i], label="n={}".format(n))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
