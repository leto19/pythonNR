import numpy as np
import math
from estimate_snrs import estimate_snrs
from   init_noise_tracker_ideal_vad import init_noise_tracker_ideal_vad
from  lookup_gains_in_table import lookup_gain_in_table

def algorithm(noisy, parameters):
    # where noisy is an ndarray of audio and parameters is a dictionary
    MIN_GAIN = parameters['min_gain']
    ALPHA = parameters['alpha']
    SNR_LOW_LIM = parameters['snr_low_lim'] 
    g_mag = parameters['g_mag']
    frLen = parameters['frLen']
    fShift = parameters['fShift']
    nFrames = math.floor(len(noisy) / fShift) - 1

    fs = parameters['fs']
    anWin = parameters['anWin']
    synWin = parameters['synWin']
    fft_size = frLen
    eps = 2.2204e-16
    noise_psd = init_noise_tracker_ideal_vad(
        noisy, frLen, frLen, fShift, anWin)
    # print("n_psd:",noise_psd)
    noise_psd = np.maximum(eps, noise_psd)

    PH1mean = 0.5
    alphaPH1mean = 0.9
    alphaPSD = 0.8

    q = 0.5
    priorFact = q / (1 - q)
    xiOptDb = 15
    xiOpt = np.power(10, (xiOptDb / 10))
    logGLRFact = np.log(1 / (1 + xiOpt))
    GLRexp = xiOpt / (1 + xiOpt)
    clean_est_dft_frame = []

    shat = np.zeros(np.size(noisy))
    #g_mag = np.round(g_mag,decimals=4)
    for indFr in range(1, nFrames):
        indices = np.arange(int((indFr - 1) * fShift), int((indFr - 1) * fShift + frLen))
        noisy_frame = anWin * noisy[indices]

        noisyDftFrame = np.fft.fft(noisy_frame, frLen)
        noisyDftFrame = noisyDftFrame[np.arange(0, int(frLen / 2)+1)]

        clean_est_dft_frame_p = np.power(np.abs(clean_est_dft_frame), 2)
        noisyPer = noisyDftFrame*(np.conj(noisyDftFrame))

        snrPost1 = noisyPer / (noise_psd)

        # Noise Power Estimation
        GLR = priorFact * np.exp(np.minimum(logGLRFact + np.dot(GLRexp,snrPost1), 200))
        PH1 = GLR / (1 + GLR)


        PH1mean = alphaPH1mean * PH1mean + (1 - alphaPH1mean) * PH1
        stuckInd = PH1mean > 0.99
        PH1[stuckInd] = np.minimum(PH1[stuckInd], 0.99)
        estimate = PH1 * noise_psd + (1 - PH1) * noisyPer
        noise_psd = alphaPSD * noise_psd + (1 - alphaPSD) * estimate

        a_post_snr, a_priori_snr = estimate_snrs(
            noisyPer, fft_size, noise_psd, SNR_LOW_LIM, ALPHA, indFr, clean_est_dft_frame_p)
        gain = lookup_gain_in_table(
            g_mag, a_post_snr, a_priori_snr, np.arange(-40, 51, 1), np.arange(-40, 51, 1), 1)

        gain = np.maximum(gain, MIN_GAIN)
        clean_est_dft_frame = gain*noisyDftFrame[0:int(fft_size/2)+1]
        clean_conj_flip = np.flipud(np.conj(clean_est_dft_frame[1:-1]))
        to_concat = np.concatenate((clean_est_dft_frame,clean_conj_flip))
        shat[indices] = shat[indices] + synWin * np.real(np.fft.ifft(to_concat))

    return shat

