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


dataset_path = "/home/zhang_muxin/DATASETS2P_result/scene_nomask_2"
output_root = "/home/zhang_muxin/DATASETS2P_result/scene_nomask_2_select"
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
            
            # --- 开始修改：使用零填充来确保形状一致 ---

            # 1. 定义窗口大小和获取原始形状
            window_size = 9
            half_window = 4  # window_size // 2
            
            original_shape = signal.shape
            num_bins_original = original_shape[-1] # 96
            other_dims = original_shape[:-1] # (2880,)

            # 2. 创建一个全零的“画布”作为输出
            # 目标形状是 (2880, 9)
            output_shape = (*other_dims, window_size) 
            sliced_signal = np.zeros(output_shape, dtype=signal.dtype)

            # 3. 找到 max_bin
            _, max_bin = get_max_bin(signal)

            # 4. 计算需要复制的源 (src) 和目标 (dest) 索引
            
            # 源：在 96-bin 信号中的有效索引
            src_start = max(0, max_bin - half_window)
            src_end = min(num_bins_original, max_bin + half_window + 1)
            
            # 目标：在 9-bin 画布中的粘贴位置
            dest_start = src_start - (max_bin - half_window)
            dest_end = dest_start + (src_end - src_start)

            # 5. 从原始信号复制数据到画布
            #    例如: max_bin = 1 (靠近边缘)
            #    src_start=0, src_end=6
            #    dest_start=3, dest_end=9
            #    这会将 signal[..., 0:6] 复制到 sliced_signal[..., 3:9]
            #    最终 sliced_signal 的形状仍然是 (2880, 9)，前 3 个 bin 为 0
            sliced_signal[..., dest_start:dest_end] = signal[..., src_start:src_end]
            
            # 6. 保存这个固定形状的数组
            data_to_save = {'data': sliced_signal}
            savemat(output_path, data_to_save)
            
            # --- 结束修改 ---

        except Exception as e:
            print(f"!!! Error processing file: {dat_path}")
            print(f"!!! ERROR: {e}")
            import traceback
            traceback.print_exc() # 打印更详细的错误
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
    process_category("扳手")
    process_category("笔记本电脑")
    process_category("刀具")
    process_category("键盘")
    process_category("手机")
    process_category("水杯")
    process_category("显示器")
    process_category("易拉罐")
    
