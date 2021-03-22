from TabulateGainGamma1 import TabulateGainGamma1
from TabulateGenGam1dft import TabulateGenGam1dft
import numpy as np
def tabulate_gain_functions(gamma, nu):

    if gamma==2:
        if nu <=0:
            return "error!"
        else:
            pass
    elif gamma==1:
        if nu <= 0.5:
            return "error!"
        else:
            K = 20
            Gdft = TabulateGenGam1dft(np.arange(-40, 51),
                                      np.arange(-40, 51), nu, K)
            Gmag, Gmag2 = TabulateGainGamma1(
                np.arange(-40, 51), np.arange(-40, 51), nu, K)
    elif (gamma != 1 or gamma!=2):
        return "error!"
    return Gdft, Gmag, Gmag2