from scipy.io import *
import numpy as np
def useful_range_bin_localization(signal:np.ndarray):
    signal_abs = np.abs(signal)
    frame_num = signal.shape[0]
    bins_num = signal.shape[1]
    useful_bins_num = 9
    left_bin_index = 0
    right_bin_index = useful_bins_num-1
    amplitude_sum = np.sum(signal_abs[:,left_bin_index:right_bin_index+1])
    max_amplitude_sum = amplitude_sum
    target_left_bin_index = left_bin_index
    target_right_bin_index = right_bin_index
    while right_bin_index<bins_num-1:
        right_bin_index+=1
        amplitude_sum += np.sum(signal_abs[:,right_bin_index])
        amplitude_sum -= np.sum(signal_abs[:,left_bin_index])
        left_bin_index+=1
        if amplitude_sum>max_amplitude_sum:
            max_amplitude_sum = amplitude_sum
            target_left_bin_index = left_bin_index
            target_left_bin_index = right_bin_index
    return signal[:,target_left_bin_index:target_right_bin_index+1]
        
