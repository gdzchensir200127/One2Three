'''
get the top 9 bins with max strength
'''

import sys
sys.path.append('/home/zhang_muxin')
import os, multiprocessing
import warnings
import open3d as o3d
from plot import get_max_bin, plot_radar_data
import numpy as np
from scipy.io import loadmat, savemat  
import matplotlib.pyplot as plt
from tqdm import tqdm


dataset_path = "/home/zhang_muxin/DATASETS2P_original/data_mat"
output_root = "/home/zhang_muxin/DATASETS2P_original/data_mat_select"
mat_names = ["0.mat", "1.mat", "2.mat", "3.mat"]

def process_file(dir_path: str):
    for mat_name in mat_names:
        try:
            dat_path = os.path.join(dir_path, mat_name)
            output_path = dat_path.replace(dataset_path, output_root)
            output_path_dir = os.path.dirname(output_path)
            if not os.path.exists(output_path_dir):
                os.makedirs(output_path_dir)

            signal = loadmat(dat_path)['data']  # 原始形状 (2880, 96)
            
            # --- 开始修改：直接固定截取 11-20 (共9个bin) ---
            
            # Python 切片是 "左闭右开" 的
            # [11:20] 表示取索引 11, 12, 13, 14, 15, 16, 17, 18, 19 
            # 刚好是 9 个数据
            sliced_signal = signal[..., 11:20]
            
            # --- 结束修改 ---

            # 保存
            data_to_save = {'data': sliced_signal}
            savemat(output_path, data_to_save)
            
        except Exception as e:
            print(f"!!! Error processing file: {dat_path}")
            print(f"!!! ERROR: {e}")
            import traceback
            traceback.print_exc() 
            continue

def process_category(category: str):
    root_dir = os.path.join(dataset_path, category)
    file_list = []
    for dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, dir)):
            
            # 直接使用 dir 作为处理路径
            obj_path = os.path.join(root_dir, dir)
            
            file_list.append(obj_path)
            
    with tqdm(total=len(file_list), desc='Processing files') as pbar:
        pool = multiprocessing.Pool(processes=20)
        for _ in pool.imap_unordered(process_file, file_list):
            pbar.update(1)

if __name__ == "__main__":
    process_category("扳手-遮挡")
    process_category("笔记本电脑-遮挡")
    process_category("刀具-遮挡")
    process_category("键盘-遮挡")
    process_category("手机-遮挡")
    process_category("水杯-遮挡")
    process_category("显示器-遮挡")
    process_category("易拉罐-遮挡")
    
