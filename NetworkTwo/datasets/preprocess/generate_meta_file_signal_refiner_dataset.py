import os,sys
import numpy as np
import cv2
import trimesh
import sklearn.preprocessing
import multiprocessing
from tqdm import tqdm
import pickle
import random
dataset_path = "/home/zhang_muxin/Signal2PC/datasets/data/signal_refine/real_dataset_simulate_signal_aug"
meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/signal_refine/meta"
mat_names = ["0.mat","1.mat","2.mat","3.mat" ]
train_all_list = []
test_all_list = []
all_list = []

def process_category(category:str):
    global train_all_list, test_all_list, all_list
    root_dir = os.path.join(dataset_path,category)
    train_meta_file = os.path.join(meta_dir,"train_" + category + "_signal_refiner_dataset.txt")
    test_meta_file = os.path.join(meta_dir,"test_" + category + "_signal_refiner_dataset.txt")
    file_list = []
    for dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir,dir)):
            real_dir = os.path.join(root_dir,dir,'models')
            vaild = True
        for i in range(4):
            mat_name = mat_names[i]
            mat_path = os.path.join(real_dir,mat_name)
            if os.path.exists(os.path.join(real_dir,mat_name)):
                real_mat_path = os.path.join(real_dir,mat_name)
                file_size = os.path.getsize(real_mat_path)/1024
                if file_size < 390:
                    vaild = False
                    break
            else:
                vaild = False
                break
        if vaild:
            for i in range(4):
                mat_name = mat_names[i]
                mat_path = os.path.join(real_dir,mat_name)
                file_list.append(mat_path)
                all_list.append(mat_path)

    random.shuffle(file_list)

    train_len = len(file_list) // 5 * 4
    
    with open(train_meta_file, 'w') as file:
        for item in file_list[: train_len]:
            train_all_list.append(item)
            file.write(item + '\n')
    with open(test_meta_file, 'w') as file:
        for item in file_list[train_len:]:
            test_all_list.append(item)
            file.write(item + '\n')

def process_all():
    global all_list
    train_meta_file = os.path.join(meta_dir,"train_all_signal_refiner_dataset.txt")
    test_meta_file = os.path.join(meta_dir,"test_all_signal_refiner_dataset.txt")

    random.shuffle(all_list)
    train_len = len(all_list) // 5 * 4
    with open(train_meta_file, 'w') as file:
        for item in all_list[: train_len]:
            file.write(item + '\n')
    with open(test_meta_file, 'w') as file:
        for item in all_list[train_len :]:
            file.write(item + '\n')

if __name__=="__main__":
    process_category("笔记本电脑")
    process_category("显示器")
    process_category("手机")
    process_category("键盘")
    process_category("水杯")
    process_category("刀具")
    process_category("扳手")
    process_all()
