import os,sys
import numpy as np
import cv2
import trimesh
import sklearn.preprocessing
import multiprocessing
from tqdm import tqdm
import pickle
dataset_path = "/home/zhang_muxin/shapenetS2M"
meta_dir = "/home/zhang_muxin/Signal2mesh/datasets/data/shapenet/meta/"
dat_names = ["0.dat","1.dat","2.dat","3.dat"]
mat_names = ["0.mat","1.mat","2.mat","3.mat"]
train_all_list = []
test_all_list = []
def process_category(category:str):
  root_dir = os.path.join(dataset_path,category)
  train_meta_file = os.path.join(meta_dir,"train_tf_" + category + "_S2M.txt")
  test_meta_file = os.path.join(meta_dir,"test_tf_" + category + "_S2M.txt")
  file_list = []
  for dir in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir,dir)):
      real_dir = os.path.join(root_dir,dir,'models')
      dir_path = os.path.join(root_dir,dir,'models').replace(dataset_path,'Data/ShapeNetP2M')
      for i in range(4):
        dat_name = dat_names[i]
        mat_name = mat_names[i]
        dat_path = os.path.join(dir_path,dat_name)
        mat_path = os.path.join(dir_path,mat_name)
        if os.path.exists(os.path.join(real_dir,mat_name)) and os.path.exists(os.path.join(real_dir,dat_name)):
          real_mat_path = os.path.join(real_dir,mat_name)
          file_size = os.path.getsize(real_mat_path)/1024
          if file_size > 390:
            file_list.append(dat_path)
  file_list.sort()
  with open(train_meta_file, 'w') as file:
    for item in file_list[:4*len(file_list)//5]:
      train_all_list.append(item)
      file.write(item + '\n')
  with open(test_meta_file, 'w') as file:
    for item in file_list[4*len(file_list)//5:]:
      test_all_list.append(item)
      file.write(item + '\n')

def process_all():
  train_meta_file = os.path.join(meta_dir,"train_tf_all_S2M.txt")
  test_meta_file = os.path.join(meta_dir,"test_tf_all_S2M.txt")
  with open(train_meta_file, 'w') as file:
    for item in train_all_list:
      file.write(item + '\n')
  with open(test_meta_file, 'w') as file:
    for item in test_all_list:
      file.write(item + '\n')

if __name__=="__main__":
  process_category("03211117")
  process_category("03636649")
  process_category("03691459")
  process_category("04090263")

  process_category("04256520")

  process_category("04401088")

  process_category("04379243")

  process_category("02691156")
  
  process_category("04530566")
  process_category("02828884")
  process_category("02933112")

  # # process_category("02958343")
  # # process_category("03001627")
  process_all()
