'''
use the obj file after HPR to generate simulate signal
'''
from signal_simulate.simulate.simulate_moving_object_cupy1 import simulate
import cupy as cp
import multiprocessing
dataset_path = "/home/zhang_muxin/shapenetS2M_HPR"

output_root = "/home/zhang_muxin/shapenetS2M_simulate"

obj_names = ["0.obj","1.obj","2.obj","3.obj"]


import os
from tqdm import tqdm
import time

def process_file(dir_path:str):
  # device_id = 2 
  # device = cp.cuda.Device(device_id)
  # device.use()
  for obj_name in obj_names:
    input_path = os.path.join(dir_path,obj_name)

    output_path = input_path.replace(dataset_path,output_root).replace('.obj','.mat')

    output_path_dir = os.path.dirname(output_path)
    if not os.path.exists(output_path_dir):
      os.makedirs(output_path_dir)
    # if os.path.exists(output_path):
    #   continue
    simulate(input_path,output_path)

def process_category(category:str):
  print(category)
  root_dir = os.path.join(dataset_path,category)
  file_list = []
  for dir in os.listdir(root_dir):
      if os.path.isdir(os.path.join(root_dir,dir)):
        obj_path = os.path.join(root_dir,dir,'models')
        file_list.append(obj_path)
  file_list.sort()
  # file_list.reverse()
  # file_list = file_list[150:]
  with tqdm(total=len(file_list), desc='Processing files') as pbar:
    pool = multiprocessing.Pool(processes=10)
    for _ in pool.imap_unordered(process_file, file_list):
      pbar.update(1)
    # for file in file_list:
    #   process_file(file)
    #   pbar.update(1)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    # process_category("扳手")
    # process_category("笔记本电脑")
    # process_category("刀具")
    # process_category("键盘")
    # process_category("手机")
    # process_category("水杯")
    # process_category("显示器")
    # process_category("03211117")
    # process_category("03636649")
    # process_category("03691459")
    process_category("04090263")

    process_category("04256520")

    process_category("04401088")

    process_category("04379243")

    process_category("02691156")

    process_category("04530566")
    process_category("02828884")
    process_category("02933112")

    # process_category("02958343")
    # process_category("03001627")