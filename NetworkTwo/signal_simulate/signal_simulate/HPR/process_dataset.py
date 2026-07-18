import os,multiprocessing
import warnings
import open3d as o3d
from signal_simulate.HPR.hidden_faces_removal import hidden_faces_removal

sem_dataset_path = "/home/muxin/hdd/shapenet/sem/models"

import os
from tqdm import tqdm

def process_file(input_path):
  output_path = input_path.replace("shapenet", 'processed_shapenet')
  folder_path = os.path.dirname(output_path)
  if not os.path.exists(folder_path):
      os.makedirs(folder_path)

  if os.path.exists(output_path):
      return
  try:
    hidden_faces_removal(input_path, output_path,False)
  except Exception:
    with open('err.err','a') as f:
      print("Err in file: ",input_path,file=f)

def process_sem_dataset(root_folder):
  file_list = []
  for root, dirs, files in os.walk(root_folder):
      for file in files:
          if file.endswith('.obj'):
              file_list.append(os.path.join(root,file))
  pool = multiprocessing.Pool(12)
  with open("log.log",'w') as f:
    with tqdm(total=len(file_list), desc='Processing files',file=f) as pbar:
      for _ in pool.imap_unordered(process_file, file_list):
          pbar.update(1)

                

if __name__ == "__main__":
    process_sem_dataset(sem_dataset_path)