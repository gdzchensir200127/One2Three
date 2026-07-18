from scipy.io import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import os

def get_diff_x_rawdata(root_dir:str,exp_object:str,x_list:list):
  raw_datas = {}
  for x in x_list:
    dir_path = os.path.join(root_dir,exp_object,x)
    files = os.listdir(dir_path)
    files = sorted(files)
    
    file_path = os.path.join(dir_path,files[0])

    data = loadmat(file_path)['data']

    raw_datas[x] = data
  
  return raw_datas

def plot_radar_data(ax:axes.Axes,raw_data:np.ndarray,title:str):
  c = ax.imshow(np.abs(raw_data), cmap='viridis', aspect='auto', origin='lower')

  cbar = plt.colorbar(c)
  cbar.set_label('Value')

  ax.set_xlabel('Distance')
  ax.set_ylabel('Time')
  
  ax.set_title(title)

def get_max_bin(data:np.ndarray):
  '''
  get the signal of the bin which has the max strength
  data: [frames * bins]
  '''

  strength_sum = np.abs(np.sum(data,axis=0))
  max_index = np.argmax(strength_sum)

  return data[:,max_index].reshape(data.shape[0]),max_index



def plot_iq_data(ax:axes.Axes,bin_data:np.ndarray,color:str='r',text:str='',title:str='iq data'):
  real_part = np.real(bin_data)
  imag_part = np.imag(bin_data)

  ax.scatter(real_part,imag_part,color=color,marker = '.')

  ax.text(real_part[0],imag_part[0],text,fontsize=15)

  ax.set_title(title)

def plot_real_part(ax:axes.Axes,bin_data:np.ndarray,title:str='real part of signal'):
  real_part = np.real(bin_data)

  ax.plot(real_part)

  ax.set_title(title)

def plot_imag_part(ax:axes.Axes,bin_data:np.ndarray,title:str='imag part of signal'):
  imag_part = np.imag(bin_data)

  ax.plot(imag_part)

  ax.set_title(title)

def plot_phase(ax:axes.Axes,bin_data:np.ndarray,title:str='phase of signal'):
  phase = np.angle(bin_data)

  ax.plot(phase)

  ax.set_title(title)


