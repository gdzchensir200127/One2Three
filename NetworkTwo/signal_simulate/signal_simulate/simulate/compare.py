from scipy.io import *
import numpy as np
import matplotlib.pyplot as plt
import os
from signal_simulate.util.plot import *

real_data_root = "/home/muxin/hdd/signal_simulate/data/real_data"
simulate_data_root = "/home/muxin/hdd/signal_simulate/data/simulate_data"



if __name__ == "__main__":
  # data_reference_path = os.path.join(real_data_root,"reference/20231123_143121.mat")
  # real_data_path = os.path.join(real_data_root,"cylinder/20231123_142910.mat")
  # simulate_data_path = os.path.join(simulate_data_root,"cylinder_63.mat")

  data_reference_path = os.path.join(real_data_root,"reference/20231118_165556.mat")
  real_data_path = os.path.join(real_data_root,"display/20231118_165301.mat")
  simulate_data_path = os.path.join(simulate_data_root,"display_60_cp.mat")
  
  data_reference = loadmat(data_reference_path)['data'][0:2800,:]

  real_data = loadmat(real_data_path)['data'][0:2800,:]


  simulate_data = loadmat(simulate_data_path)['data']
  simulate_data = simulate_data[0:2800,5:101]

  simulate_data_max_bin,simulate_data_max_bin_index = get_max_bin(simulate_data)
  print(f'max bin index of simulate data: {simulate_data_max_bin_index}')
  real_data = real_data - data_reference

  real_data_max_bin,real_data_max_bin_index = get_max_bin(real_data)
  print(f'max bin index of real data: {real_data_max_bin_index}')
  fig,ax = plt.subplots(1,2,figsize = (18,12))

  plot_radar_data(ax[0],real_data,"real data")

  plot_radar_data(ax[1],simulate_data,"simulate data")

  fig_iq,ax_iq = plt.subplots(1,2,figsize = (18,12))

  plot_iq_data(ax_iq[0],real_data_max_bin,title='iq of the max bin of the real data')
  plot_iq_data(ax_iq[1],simulate_data_max_bin,title='iq of the max bin of the simulate data')

  fig_real,ax_real = plt.subplots(2,1,figsize = (18,12))

  plot_real_part(ax_real[0],real_data_max_bin,title='real part of the max bin of the real data')
  plot_real_part(ax_real[1],simulate_data_max_bin,title='real part of the max bin of the simulate data')

  fig_imag,ax_imag = plt.subplots(2,1,figsize = (18,12))

  plot_imag_part(ax_imag[0],real_data_max_bin,title='imag part of the max bin of the real data')
  plot_imag_part(ax_imag[1],simulate_data_max_bin,title='imag part of the max bin of the simulate data')

  fig_phase,ax_phase = plt.subplots(2,1,figsize = (18,12))

  plot_phase(ax_phase[0],real_data_max_bin,title='phase of the max bin of the real data')
  plot_phase(ax_phase[1],simulate_data_max_bin,title='phase of the max bin of the simulate data')

  plt.show()