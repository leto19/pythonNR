import numpy as np

from scipy.special import gamma
import transplant
from ParCylFun import ParCylFun

def TabulateGenGam1dft(Rksi, Rgam, nu, K):
    Rgam = np.power(10, (Rgam / 10))  # in dBs
    Rksi = np.power(10, (Rksi / 10))
    # matlab = transplant.Matlab()
    # matlab.addpath("/home/george/matlab_projects/code_nr_alg3_book/TabGenGam/")
    G = np.zeros((len(Rksi), len(Rgam)))
    for k in range(1, len(Rksi) + 1):
        ksi = Rksi[k - 1]

        Glow, Glow2 = lowSNRgains(nu, Rgam, ksi, K)
        Ghigh, Ghigh2 = highSNRgains(nu, Rgam, ksi)
        G[k - 1, :] = np.maximum(Glow, Ghigh2)
    print(G)
    return G


def lowSNRgains(nu, gamm, ksi, K):
    alpha = nu
    mu = np.sqrt(alpha * (alpha + 1))

    gamm = gamm.flatten()
    ksi = ksi.flatten()
    Lg = len(gamm)
    Lk = len(ksi)

    x = mu / np.sqrt(2 * ksi)

    FACK2 = np.zeros((K, 1))
    FACK2[K - 1] = 1 / gamma(K) ** 2
    GAMMA2KA = np.zeros((2 * K + 1, 1))
    GAMMA2KA[2 * K] = gamma(alpha + 2 * K)
    GAMMA2KA[2 * K - 1] = GAMMA2KA[2 * K] / (alpha + 2 * (K - 1))
    GAMMA2KA[2 * K - 2] = GAMMA2KA[2 * K - 1] / (alpha + 2 * K - 2)

    GAMM2K = np.zeros((K, Lg))
    GAMM2K[K - 1, :] = np.power((gamm / 2), (K - 1))
    D = np.zeros((2 * K + 1, Lk))


    D[2 * K, :] = ParCylFun(-alpha - 2 * K, x)[0]
    D[2 * K - 1, :] = ParCylFun(-alpha - 2 * K + 1, x)[0]
    D[2 * K - 2, :] = (alpha + 2 * K - 1) * D[2 * K, :] + x * D[2 * K - 1, :]

    """
    D[2*K,:] = 1.5670e-75  # fix this later
    D[2*K-1,:] = 1.0947e-73
    D[2*K-2,:] = (alpha+2*K-1)*D[2*K,:]+x*D[2*K-1,:]
    """
    T1 = FACK2[K - 1] * GAMM2K[K - 1, :] * np.dot(GAMMA2KA[2 * K - 1, :], D[2 * K, :])

    T2 = (1 / (np.math.factorial(K - 1) * np.math.factorial(K))) * GAMM2K[K - 1, :] * GAMMA2KA[2 * K, :] * (D[2 * K, :])
    T3 = FACK2[K - 1] * GAMM2K[K - 1, :] * np.dot(GAMMA2KA[2 * K], D[2 * K, :])
    N1 = FACK2[K - 1] * GAMM2K[K - 1, :] * GAMMA2KA[2 * K - 2] * D[2 * K - 2, :]

    for k in range(K - 1, 0, -1):
        fack2 = FACK2[k] * np.power(k, 2)
        FACK2[k - 1] = fack2

        gamm2k = GAMM2K[k, :] / (gamm / 2)
        GAMM2K[k - 1, :] = gamm2k

        even2ka = GAMMA2KA[2 * k] / (alpha + 2 * k - 1)
        GAMMA2KA[2 * k - 1] = even2ka

        odd2ka = even2ka / (alpha + 2 * k - 2)
        GAMMA2KA[2 * k - 2] = odd2ka

        Deven = (alpha + 2 * k) * D[2 * k + 1, :] + x * D[2 * k, :]
        D[2 * k - 1, :] = Deven

        Dodd = (alpha + 2 * k - 1) * D[2 * k, :] + x * Deven
        D[2 * k - 2, :] = Dodd

        T2 = T2 + (1 / (np.math.factorial(k - 1) * np.math.factorial(k))) * gamm2k * GAMMA2KA[2 * k] * D[2 * k, :]
        T3 = T3 + fack2 * gamm2k * np.dot(GAMMA2KA[2 * k], D[2 * k])
        T1 = T1 + fack2 * gamm2k * np.dot(even2ka, Deven)
        N1 = N1 + fack2 * gamm2k * odd2ka * Dodd

    G = T2 / N1
    G = G / 2
    Gmeth2 = ((mu / (np.sqrt(8 * ksi * np.power(gamm, 2)))) * (T1 / N1) + ((1. / (2 * gamm)) * (T3 / N1)) - (
                alpha / (2 * gamm)))
    G = G.T
    Gmeth2 = Gmeth2.T
    return G, Gmeth2


def highSNRgains(nu, gamm, ksi):
    alpha = nu
    mu = np.sqrt(alpha * (alpha + 1))
    x = mu / np.sqrt(2 * ksi) - np.sqrt(2 * gamm)
    # print(alpha,x)
    # print(type(alpha),type(x))
    # print(np.shape(x))
    # matlab = transplant.Matlab()
    # matlab.addpath("/home/george/matlab_projects/code_nr_alg3_book/TabGenGam/")
    pdf = ParCylFun(-alpha - 1.5, x)[0]

    Dteller2 = pdf

    pdf = ParCylFun(-alpha - 0.5, x)[0]

    Dteller = pdf
    Dnoemer = (alpha + 0.5) * Dteller2 + x * Dteller
    G = (alpha - 0.5) / np.sqrt(2 * gamm) * Dteller / Dnoemer
    Gmeth2 = (((alpha - 0.5) * mu / np.sqrt(8 * ksi * gamm ** 2))) * (Dteller / Dnoemer) + \
             (((alpha + 0.5) * (alpha - 0.5) / (2 * gamm)) * (Dteller2 / Dnoemer)) - (alpha / (2 * gamm))
    # G = G.T
    # Gmeth2 = Gmeth2.T
    return G, Gmeth2
