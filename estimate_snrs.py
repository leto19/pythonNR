
import numpy as np

def estimate_snrs(noisy_dft_frame_p, fft_size, noise_psd, SNR_LOW_LIM, ALPHA, I, clean_est_dft_frame_p):
    a_post_snr = noisy_dft_frame_p / (noise_psd[0:fft_size])
    if I == 1:
        a_priori_snr = np.maximum(a_post_snr-1, SNR_LOW_LIM)
    else:
        a_priori_snr = np.maximum(ALPHA*(clean_est_dft_frame_p[0:int(fft_size/2)+1]) /
                           noise_psd[0:int(fft_size/2)+1]+(1-ALPHA)*(a_post_snr-1), SNR_LOW_LIM)

    return a_post_snr, a_priori_snr