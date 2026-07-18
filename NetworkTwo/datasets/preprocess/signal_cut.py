'''
get the top 9 bins with max strength
'''
import os,multiprocessing
import warnings
import open3d as o3d
from signal_simulate.util.plot import get_max_bin,plot_radar_data
import numpy as np
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from tqdm import tqdm

dataset_path = "/home/zhang_muxin/shapenetS2M_simulate"

output_root = "/home/zhang_muxin/shapenetS2M_simulate_cut"

mat_names = ["0.mat","1.mat","2.mat","3.mat"]

def process_file(dir_path:str):
    for mat_name in mat_names:
        try:
          dat_path = os.path.join(dir_path,mat_name)
          output_path = dat_path.replace(dataset_path,output_root)
          output_path_dir = os.path.dirname(output_path)
          if not os.path.exists(output_path_dir):
              os.makedirs(output_path_dir)

          if os.path.exists(output_path):
            continue

          signal = loadmat(dat_path)['data']
          # fig,ax = plt.subplots(1,1)

          # plot_radar_data(ax,signal,'init signal')

          # plt.show()

          # _,max_bin = get_max_bin(signal)
          # start_bin = max_bin-4
          # end_bin = max_bin+4
          start_bin = 12
          end_bin = 20
          signal = signal[:,start_bin:end_bin+1]

          # fig2,ax2 = plt.subplots(1,1)
          # plot_radar_data(ax2,signal,'signal after cut')

          # plt.show()
          
          savemat(output_path,{'data':signal})
        except:
           continue

def process_category(category:str):
  root_dir = os.path.join(dataset_path,category)
  file_list = []
  for dir in os.listdir(root_dir):
      if os.path.isdir(os.path.join(root_dir,dir)):
        obj_path = os.path.join(root_dir,dir,'models')
        file_list.append(obj_path)
  with tqdm(total=len(file_list), desc='Processing files') as pbar:
    pool = multiprocessing.Pool(processes=20)
    for _ in pool.imap_unordered(process_file, file_list):
      pbar.update(1)
      
if __name__ == "__main__":
  # process_category("03211117")
  # process_category("03636649")
  # process_category("03691459")
  # process_category("04090263")

  # process_category("04256520")

  # process_category("04401088")

  # process_category("04379243")

  # process_category("02691156")
  
  # process_category("04530566")
  # process_category("02828884")
  # process_category("02933112")

  # # process_category("02958343")
  # # process_category("03001627")
  process_category("扳手")
  process_category("笔记本电脑")
  process_category("刀具")
  process_category("键盘")
  process_category("手机")
  process_category("水杯")
  process_category("显示器")