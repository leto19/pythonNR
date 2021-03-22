import numpy as np
import transplant
from scipy.special import gamma as get_gamma
from ParCylFun import ParCylFun
def TabulateGainGamma1(Rprior,Rpost,nu,K):
    Rpost = np.power(10,(Rpost/10))
    Rprior=np.power(10,(Rprior/10))
    G1 = np.zeros((len(Rprior),len(Rpost)))
    G2 = np.copy(G1)

    # matlab = transplant.Matlab()
    # matlab.addpath("/home/george/matlab_projects/code_nr_alg3_book/TabGenGam/")
    for k in range(1,len(Rprior)+1):
        SNRprior = Rprior[k-1]
        Glow,G2low = lowSNRgains(nu,Rpost,SNRprior,K)
        Ghigh,G2high = highSNRgains(nu,Rpost,SNRprior)
        G1[k-1, :] = np.maximum(Glow,Ghigh)
        G2[k-1,:] = np.maximum(G2low,G2high)
    return G1,G2

def lowSNRgains(nu,SNRpost,SNRprior,K):
    mu = np.sqrt(nu*(nu+1))

    SNRpost = np.transpose(SNRpost)
    SNRprior = np.transpose(SNRprior)
    Lg = len(SNRpost)
    Lk = np.size(SNRprior)

    x = mu/np.sqrt(2*SNRprior)

    FACK2 = np.zeros((K,1))
    FACK2[K-1] = 1/get_gamma(K)**2
    GAMMA2KA = np.zeros((2*K+1,1))
    GAMMA2KA[2*K] = get_gamma(nu+2*K)
    GAMMA2KA[2*K-1] = GAMMA2KA[2*K]/(nu+2*K-1)
    GAMMA2KA[2 * K - 2] = GAMMA2KA[2 * K-1] / (nu + 2 * K - 2)
    SNRPOST2K = np.zeros((K,Lg))
    SNRPOST2K[K-1,:] = np.power((SNRpost/2), (K-1))
    D = np.zeros((2*K+1,Lk))
    D[2*K,:] = ParCylFun(-nu-2*K,x)[0]
    D[2*K-1,:] = ParCylFun(-nu-2*K+1,x)[0]
    D[2*K-2,:] = (nu+2*K-2)*D[2*K,:] + x*D[2*K-1,:]

    Num1 = FACK2[K-1] * SNRPOST2K[K-1, :] * GAMMA2KA[(2 * K)-1] * D[2 * K-1, :]
    Num2 = FACK2[K-1] * SNRPOST2K[K-1, :] * GAMMA2KA[2 * K ] * D[2 * K, :]
    Den1 = FACK2[K-1] * SNRPOST2K[K-1, :] * GAMMA2KA[2 * K - 2] * D[2 * K - 2, :]

    for k in range(K-1,0,-1):

        fack2 = FACK2[k] * np.power(k,2)
        FACK2[k-1] = fack2

        SNRpost2k = SNRPOST2K[k, :] / (SNRpost / 2)
        SNRPOST2K[k-1, :] = SNRpost2k

        even2ka = GAMMA2KA[2 * k] / (nu + 2 * k - 1)
        GAMMA2KA[2 * k] = even2ka

        odd2ka = even2ka / (nu + 2 * k - 2)
        GAMMA2KA[2 * k - 2] = odd2ka

        Deven = (nu + 2 * k) * D[2 * k + 1, :] + x * D[2 * k, :]
        D[2 * k-1, :] = Deven

        Dodd = (nu + 2 * k - 1) * D[2 * k, :] + x * Deven
        D[2 * k - 2, :] = Dodd


        Num2 = Num2 + fack2 * SNRpost2k * GAMMA2KA[2 * k] * D[2 * k, :]
        Num1 = Num1 + fack2 * SNRpost2k * even2ka * Deven
        Den1 = Den1 + fack2 * SNRpost2k * odd2ka * Dodd
    G1 = (Num1/Den1)/np.sqrt(2*SNRpost)
    G2 = (Num2/Den1)/2/SNRpost
    G1 = G1.T
    G2 = G2.T
    return G1,G2

def highSNRgains(nu,SNRpost,SNRprior):
    mu = np.sqrt(nu*(nu+1))

    x = mu/np.sqrt(2*SNRprior) - np.sqrt((2*SNRpost))

    Dnum2 = ParCylFun(-nu-1.5,x)[0]
    Dnum = ParCylFun(-nu-0.5,x)[0]

    Dden = (nu+0.5)*Dnum2+x*Dnum
    G1 = (nu-0.5)/np.sqrt(2*SNRpost)* (Dnum/Dden)
    G2 = (nu**2 -1/4)/2/SNRpost * (Dnum2/Dden)
    #G1 = G1.T
    #G2 = G2.T
    return G1,G2