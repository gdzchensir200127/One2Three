import os,multiprocessing
import warnings
import open3d as o3d
from signal_simulate.HPR.hidden_faces_removal import hidden_faces_removal
import numpy as np


dataset_path = "/home/zhang_muxin/shapenetS2M_obj"

output_root = "/home/zhang_muxin/shapenetS2M_HPR"

obj_names = ["0.obj","1.obj","2.obj","3.obj"]


import os
from tqdm import tqdm
import time

def process_file(dir_path:str):
  for obj_name in obj_names:
    try:
      input_path = os.path.join(dir_path,obj_name)
      mesh = o3d.io.read_triangle_mesh(input_path)

      output_path = input_path.replace(dataset_path,output_root)

      output_path_dir = os.path.dirname(output_path)
      if not os.path.exists(output_path_dir):
        os.makedirs(output_path_dir)
      camera = [0,0.74,0.10]
      hidden_faces_removal(input_path,output_path,False,camera)
    except:
      pass

def process_category(category:str):
  root_dir = os.path.join(dataset_path,category)
  file_list = []
  for dir in os.listdir(root_dir):
      if os.path.isdir(os.path.join(root_dir,dir)):
        obj_path = os.path.join(root_dir,dir,'models')
        file_list.append(obj_path)
  file_list = file_list[:1500]
  with tqdm(total=len(file_list), desc='Processing files') as pbar:
    pool = multiprocessing.Pool(processes=20)
    for _ in pool.imap_unordered(process_file, file_list):
      pbar.update(1)
                
if __name__ == "__main__":
  process_category("03211117")
  process_category("03636649")
  process_category("03691459")
  process_category("04090263")
  process_category("04256520")
  process_category("04379243")
  process_category("04401088")
  process_category("04530566")
  process_category("02691156")
  process_category("02828884")
  process_category("02933112")
  process_category("02958343")
  process_category("03001627")
  # process_category("扳手")
  # process_category("笔记本电脑")
  # process_category("刀具")
  # process_category("键盘")
  # process_category("手机")
  # process_category("水杯")
  # process_category("显示器")