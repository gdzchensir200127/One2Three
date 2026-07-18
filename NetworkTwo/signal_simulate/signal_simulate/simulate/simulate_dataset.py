import os,multiprocessing
import warnings
import open3d as o3d
from signal_simulate.simulate.simulate_moving_object_cupy1 import simulate

sem_dataset_path = "/home/muxin/hdd/processed_shapenet/sem/models"

import os
from tqdm import tqdm

def simulate_file(input_path):
  output_path = input_path.replace('processed_shapenet','simulate_shapenet')
  output_path = output_path.replace('.obj','.mat')
  folder_path = os.path.dirname(output_path)
  if not os.path.exists(folder_path):
      os.makedirs(folder_path)
  if os.path.exists(output_path):
      return
  try:
    simulate(input_path, output_path)
  except Exception:
    with open('err.err','a') as f:
      print("Err in file: {} with {}".format(input_path,Exception),file=f)

def simulate_sem_dataset(root_folder):
  file_list = []
  for root, dirs, files in os.walk(root_folder):
      for file in files:
          if file.endswith('.obj'):
              file_list.append(os.path.join(root,file))
  # pool = multiprocessing.Pool(1)
  with open("log.log",'w') as f:
    with tqdm(total=len(file_list), desc='Processing files') as pbar:
      for i in range(len(file_list)):
          simulate_file(file_list[i])
          pbar.update(1)

                

if __name__ == "__main__":
    simulate_sem_dataset(sem_dataset_path)