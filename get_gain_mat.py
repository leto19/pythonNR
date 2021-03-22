import transplant
import numpy as np
matlab = transplant.Matlab()

matlab.addpath("/home/george/matlab_projects/code_nr_alg3_book/TabGenGam/")

g_dft,g_mag,mag2 = matlab.Tabulate_gain_functions(1,0.6)

np.save("gain.npy",g_mag)