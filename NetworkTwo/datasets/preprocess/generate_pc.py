# -*- coding: utf-8 -*-

import os,sys
import numpy as np
import cv2
import trimesh
import sklearn.preprocessing
import multiprocessing
from tqdm import tqdm
import pickle

dataset_path = "/home/zhang_muxin/shapenetS2M_obj"

output_root = "/home/zhang_muxin/shapenetS2M"

obj_names = ["0.obj","1.obj","2.obj","3.obj"]

def process_file(dir_path:str):
    for obj_name in obj_names:
        try:
            # 1 sampling
            obj_path = os.path.join(dir_path,obj_name)
            output_path = obj_path.replace(dataset_path,output_root).replace('.obj','.dat')
            output_path_dir = os.path.dirname(output_path)
            if not os.path.exists(output_path_dir):
                os.makedirs(output_path_dir)
            if os.path.exists(output_path):
                continue
            scene = trimesh.load(obj_path)
            if isinstance(scene,trimesh.Trimesh):
                scene = trimesh.Scene(scene)
            area_sum = 0
            for _,mesh in scene.geometry.items():
                area_sum += mesh.area

            sample = np.zeros((0,3), dtype=np.float32)
            normal = np.zeros((0,3), dtype=np.float32)
            for _,mesh in scene.geometry.items():
                number = int(round(16384*mesh.area/area_sum))
                if number < 1:
                    continue
                points, index = trimesh.sample.sample_surface_even(mesh, number)
                sample = np.append(sample, points, axis=0)

                triangles = mesh.triangles[index]
                pt1 = triangles[:,0,:]
                pt2 = triangles[:,1,:]
                pt3 = triangles[:,2,:]
                norm = np.cross(pt3-pt1, pt2-pt1)
                norm = sklearn.preprocessing.normalize(norm, axis=1)
                normal = np.append(normal, norm, axis=0)
            data = np.hstack((sample, normal))
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
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
if __name__=="__main__":
  process_category("03211117")
#   process_category("03636649")
#   process_category("03691459")
#   process_category("04090263")
#   process_category("04256520")
#   process_category("04379243")
#   process_category("04401088")
#   process_category("04530566")
#   # process_category("02691156")
#   process_category("02828884")
#   process_category("02933112")
  # process_category("02958343")
  # process_category("03001627")