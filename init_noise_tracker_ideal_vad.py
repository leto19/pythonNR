import numpy as np
def init_noise_tracker_ideal_vad(noisy, fr_size, fft_size, hop, sq_hann_window):
    noisy_dft_frame_matrix = np.zeros((fft_size,5),dtype=complex)
    #  print(noisy_dft_frame_matrix.shape)
    #print(sq_hann_window)
    for i in range(1, 6):
        r = np.arange( (i-1)*hop, (i-1)*hop+fr_size)
        # print("i:%s\n"%i,r,type(r[0]))
        noisy_frame = sq_hann_window*noisy[r]
        noisy_dft_frame_matrix[:,i-1] = np.fft.fft(noisy_frame, fft_size)
    #print(noisy_frame[1])
    #print(noisy_dft_frame_matrix[0])
    #print(noisy_dft_frame_matrix.shape)
    mat_cut = noisy_dft_frame_matrix[np.arange(0,int(fr_size/2)+1)]
    #print(mat_cut.shape)
    #   print(mat_cut)
    noise_psd_init = np.mean(np.power(np.abs(mat_cut), 2), 1)
    #noise_psd_init = np.flip(noise_psd_init)  # why?
    return noise_psd_init