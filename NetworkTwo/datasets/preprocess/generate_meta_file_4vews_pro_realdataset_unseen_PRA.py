import os,sys
import numpy as np
import cv2
import trimesh
import sklearn.preprocessing
import multiprocessing
from tqdm import tqdm
import pickle
import random
dataset_path = "/home/zhang_muxin/data/augmentation_PRA"
meta_dir = "/home/zhang_muxin/data/augmentation_PRA/meta/"
dat_names = ["0.dat","1.dat","2.dat","3.dat"]
mat_names = ["0.mat","1.mat","2.mat","3.mat"]
train_all_list = []
val_all_list = []
test_all_list = []
all_list = []

def shuffle_in_blocks(lst, block_size):
    # 将列表按 block_size 分块
    blocks = [lst[i:i + block_size] for i in range(0, len(lst), block_size)]

    # 随机排列这些块
    random.shuffle(blocks)

    # 将排列后的块重新拼接成一个列表
    shuffled_list = [item for block in blocks for item in block]

    return shuffled_list

def process_category(category:str):
    global train_all_list, test_all_list, all_list
    root_dir = os.path.join(dataset_path,category)
    train_meta_file = os.path.join(meta_dir,"train_tf_" + category + "_S2M_4views_pro_realdataset_unseen_PRA.txt")
    val_meta_file = os.path.join(meta_dir,"val_tf_" + category + "_S2M_4views_pro_realdataset_unseen_PRA.txt")
    test_meta_file = os.path.join(meta_dir,"test_tf_" + category + "_S2M_4views_pro_realdataset_unseen_PRA.txt")
    file_list = []
    for dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir,dir)):
            real_dir = os.path.join(root_dir,dir,'models')
            dir_path = os.path.join(root_dir,dir,'models').replace(dataset_path,'Data/ShapeNetP2M')
            vaild = True
        for i in range(4):
            dat_name = dat_names[i]
            mat_name = mat_names[i]
            dat_path = os.path.join(dir_path,dat_name)
            mat_path = os.path.join(dir_path,mat_name)
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
                dat_name = dat_names[i]
                dat_path = os.path.join(dir_path,dat_name)
                file_list.append(dat_path)
                all_list.append(dat_path)
    file_list.sort()
    # file_list = shuffle_in_blocks(file_list, block_size=4)
    block_num = len(file_list) // 4
    train_block = block_num * 4 // 6
    val_block = block_num * 1 // 6
    test_block = block_num -val_block-train_block
    with open(train_meta_file, 'w') as file:
        for item in file_list[:4 * train_block]:
            train_all_list.append(item)
            file.write(item + '\n')
    with open(val_meta_file, 'w') as file:
        for item in file_list[4 * train_block:4 * (train_block+val_block)]:
            val_all_list.append(item)
            file.write(item + '\n')
    with open(test_meta_file, 'w') as file:
        for item in file_list[4 * (train_block+val_block):4 * (train_block+val_block+test_block)]:
            test_all_list.append(item)
            file.write(item + '\n')

def process_all():
    global all_list, train_all_list, test_all_list
    train_meta_file = os.path.join(meta_dir,"train_tf_all_S2M_4views_pro_realdataset_unseen_PRA.txt")
    val_meta_file = os.path.join(meta_dir,"val_tf_all_S2M_4views_pro_realdataset_unseen_PRA.txt")
    test_meta_file = os.path.join(meta_dir,"test_tf_all_S2M_4views_pro_realdataset_unseen_PRA.txt")
    with open(train_meta_file, 'w') as file:
        for item in train_all_list:
            file.write(item + '\n')
    with open(val_meta_file, 'w') as file:
        for item in val_all_list:
            file.write(item + '\n')
    with open(test_meta_file, 'w') as file:
        for item in test_all_list:
            file.write(item + '\n')

if __name__=="__main__":
    list = ["笔记本电脑","水杯","显示器","扳手","手机","刀具","键盘","易拉罐",
            "笔记本电脑-遮挡","水杯-遮挡","显示器-遮挡","扳手-遮挡","手机-遮挡","刀具-遮挡","键盘-遮挡","易拉罐-遮挡"]
    for category in list:
        process_category(category)
    process_all()
