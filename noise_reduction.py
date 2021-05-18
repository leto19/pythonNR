import numpy as np
from algorithm import algorithm
import soundfile as sf

def noise_reduction(in_file,out_file=None):

    if out_file == None:
        out_file = in_file.replace(".wav","_clean.wav")
    s,fs = sf.read(in_file)

    parameters = dict()
    parameters['fs'] = fs
    parameters['min_gain'] = 10**(-20/20)
    parameters['alpha'] = 0.99
    parameters['frLen'] = int(32e-3*parameters['fs'])
    parameters['fShift'] = int(parameters['frLen']/2)
    parameters['anWin'] = np.sqrt(np.hanning(parameters['frLen']))
    parameters['synWin'] = np.sqrt(np.hanning(parameters['frLen']))
    parameters['snr_low_lim'] = 2.2204e-16
    y = s
    # gamma = 1
    # nu = 0.6
    # g_dft, g_mag, g_mag2 = tabulate_gain_functions(gamma, nu)
    g_mag = np.load("gain.npy") # we don't calculate this, just load it
    parameters['g_mag'] = g_mag
    shat = algorithm(y, parameters)
    #shat = float2pcm(shat)
    #wavfile.write("out.wav",fs,shat)
    sf.write(out_file,shat,fs)
    #print("done!")

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        noise_reduction(sys.argv[1], sys.argv[2])
    else:
        noise_reduction(sys.argv[1])