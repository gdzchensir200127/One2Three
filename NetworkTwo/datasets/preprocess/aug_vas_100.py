import os
import numpy as np
import random
from scipy.io import *

# --- 路径配置 ---
root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/forS2P"
data_after_remove_background = os.path.join(root_dir, "Scene_mask")
data_after_augmentation = os.path.join(root_dir, "Scene_mask_vas")

# --- 参数配置 ---
target_frame = 2800
min_frame_offset = 0
max_frame_offset = 2880 - 2800
min_angle_offset = 0
max_angle_offset = 2 * np.pi
augmentation_factor_1 = 100
augmentation_factor = 100

def augmentation(models_dir_path, category_name, style_name, scene_name):
    """
    数据增强核心函数
    输出路径格式: Category/Style_Scene_augN/models
    """
    origin_signals = []
    
    # 1. 读取原始数据
    try:
        for i in range(4):
            mat_path = os.path.join(models_dir_path, f"{i}.mat")
            signal = loadmat(mat_path)['data']
            origin_signals.append(signal)
    except Exception as e:
        print(f"Error reading {models_dir_path}: {e}")
        return

    # 2. 生成增强参数
    augmentation_params = [[] for i in range(4)]
    for i in range(4):
        for aug_1 in range(augmentation_factor_1):
            frame_offset = random.randint(min_frame_offset, max_frame_offset)
            augmentation_params[i].append(frame_offset)
    for i in range(4):
        random.shuffle(augmentation_params[i])
    
    angle_offsets = [random.uniform(min_angle_offset, max_angle_offset) for _ in range(4)]

    # 3. 执行增强并保存
    for j in range(augmentation_factor):
        for i in range(4):
            frame_offset = augmentation_params[i][j]
            aug_signal = origin_signals[i][frame_offset:frame_offset+target_frame, :]
            rotation_factor = np.exp(1j * angle_offsets[i])
            aug_signal = aug_signal * rotation_factor
            
            # 扁平化命名: 样式_场景_aug编号 (例如: 1_14_aug0)
            new_instance_folder_name = f"{style_name}_{scene_name}_aug{j}"
            
            save_dir = os.path.join(
                data_after_augmentation, 
                category_name, 
                new_instance_folder_name,
                'models'
            )
            
            save_path = os.path.join(save_dir, f'{i}.mat')
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            savemat(save_path, {'data': aug_signal})

if __name__ == "__main__":
    if not os.path.exists(data_after_remove_background):
        print(f"Error: 输入目录不存在 {data_after_remove_background}")
        exit()
    
    # --- 定义需要处理的目标场景 ---
    target_scenes = ['14', '15', '16']

    # 1. 遍历种类
    category_list = os.listdir(data_after_remove_background)
    for category in category_list:
        category_path = os.path.join(data_after_remove_background, category)
        if not os.path.isdir(category_path): continue
        
        # 2. 遍历样式
        style_list = os.listdir(category_path)
        for style in style_list:
            style_path = os.path.join(category_path, style)
            if not os.path.isdir(style_path): continue

            # 3. 遍历场景
            scene_list = os.listdir(style_path)
            for scene in scene_list:
                
                # --- 修改点：判断是否为目标场景 ---
                if scene not in target_scenes:
                    continue  # 如果不是14, 15, 16，直接跳过
                
                scene_path = os.path.join(style_path, scene)
                if not os.path.isdir(scene_path): continue

                models_path = os.path.join(scene_path, "models")
                
                if os.path.exists(models_path):
                    print(f"Processing: {category} | Style: {style} | Scene: {scene}") # 打印日志方便确认
                    augmentation(models_path, category, style, scene)