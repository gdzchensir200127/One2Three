import os,multiprocessing
import warnings
import open3d as o3d
from signal_simulate.HPR.rotate_mesh import rotate_mesh
from signal_simulate.HPR.scale_mesh import scale_mesh_factor
import numpy as np


dataset_path = "/home/zhang_muxin/shapenetS2M_init/"

output_root = "/home/zhang_muxin/shapenetS2M_obj/"

init_obj_name = "model_normalized.obj"

scale_factor = 1

rotate_views = [(0,0,0),(0,0,np.pi/2),(0,0,np.pi),(0,0,np.pi*3/2)]

import os
from tqdm import tqdm
import time

def process_file(input_path:str):
  mtl_path = input_path.replace('obj','mtl')
  if os.path.exists(mtl_path):
    os.remove(mtl_path)
  start_time = time.time()
  mesh = o3d.io.read_triangle_mesh(input_path)
  # # v2 数据集，绕x旋转90度
  # rotation_matrix = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))  # 90 度转为弧度（π/2）

  # # 应用旋转矩阵
  # mesh.rotate(rotation_matrix)

  end_time = time.time()
  print('load time: ',end_time-start_time)

  start_time = time.time()

  scaled_mesh = scale_mesh_factor(mesh,scale_factor,np.array([0.,0.,0.]),False)

  end_time = time.time()
  print('scaled time: ',end_time-start_time)  

  for i,rotate_view in enumerate(rotate_views):
    output_path = input_path.replace(dataset_path,output_root)
    output_path = output_path.replace(init_obj_name,"%d.obj" % (i))
    output_path_dir = os.path.dirname(output_path)
    if not os.path.exists(output_path_dir):
      os.makedirs(output_path_dir)
    if os.path.exists(output_path):
      continue
    start_time = time.time()
    rotated_mesh = rotate_mesh(scaled_mesh,rotate_view,(0,0,0),False)
    end_time = time.time()
    print('rotate time: ',end_time-start_time) 
    start_time = time.time()
    o3d.io.write_triangle_mesh(output_path, rotated_mesh)
    end_time = time.time()
    print('write time: ',end_time-start_time) 

def process_category(category:str):
  root_dir = os.path.join(dataset_path,category)
  file_list = []
  for dir in os.listdir(root_dir):
      if os.path.isdir(os.path.join(root_dir,dir)):
        init_obj_path = os.path.join(root_dir,dir,'models',init_obj_name)
        file_list.append(init_obj_path)
  with tqdm(total=len(file_list), desc='Processing files') as pbar:
    for file in file_list:
      process_file(file)
      pbar.update(1)
    # pool = multiprocessing.Pool(processes=10)
    # for _ in pool.imap_unordered(process_file, file_list):
    #   pbar.update(1)
                

if __name__ == "__main__":
  # process_category("03211117")
  # process_category("03636649")
  # process_category("03691459")
  # process_category("04090263")
  # process_category("04256520")
  # process_category("04379243")
  # process_category("04401088")
  # process_category("04530566")
  # process_category("02691156")
  # process_category("02828884")
  # process_category("02933112")
  # process_category("02958343")
  # process_category("03001627")
  process_category("扳手")
  process_category("笔记本电脑")
  process_category("刀具")
  process_category("键盘")
  process_category("手机")
  process_category("水杯")
  process_category("显示器")
  pass