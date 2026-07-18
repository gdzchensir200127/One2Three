import os
import numpy as np
import random
from scipy.io import loadmat, savemat

# 更新输入输出目录路径
data_after_remove_background = "/home/zhang_muxin/Signal2PC/datasets/data/forS2P/Scene_nomask"
data_after_augmentation = "/home/zhang_muxin/Signal2PC/datasets/data/forS2Pbase/Scene_nomask"

target_frame = 2800
min_frame_offset = 0
max_frame_offset = 2880 - 2800  # 80
min_angle_offset = 0
max_angle_offset = 2 * np.pi
augmentation_factor_1 = 5
augmentation_factor_2 = 5
augmentation_factor = 25  # 5*5=25

def augmentation(instance_path):
    instance_name = os.path.basename(instance_path)
    category_name = os.path.basename(os.path.dirname(instance_path))
    
    # 遍历instance目录下的所有environment目录（新增逻辑）
    environment_list = os.listdir(instance_path)
    for environment_name in environment_list:
        env_path = os.path.join(instance_path, environment_name)
        # 跳过非目录文件（避免异常）
        if not os.path.isdir(env_path):
            print(f"Skipping non-directory file: {env_path}")
            continue
        
        # 读取当前environment下的4个信号文件（路径包含environment_name）
        origin_signals = []
        for i in range(4):
            signal_file = os.path.join(env_path, "models", f"{i}.mat")
            if not os.path.exists(signal_file):
                print(f"Warning: Signal file {signal_file} not found, skipping this environment")
                origin_signals = []
                break
            signal = loadmat(signal_file)['data']
            origin_signals.append(signal)
        
        if not origin_signals:
            continue  # 跳过缺少信号文件的environment
        
        # 生成增强参数（每个信号独立生成）
        augmentation_params = [[] for _ in range(4)]
        for i in range(4):
            for aug_1 in range(augmentation_factor_1):
                frame_offset = random.randint(min_frame_offset, max_frame_offset)
                for aug_2 in range(augmentation_factor_2):
                    angle_offset = random.uniform(min_angle_offset, max_angle_offset)
                    augmentation_params[i].append((frame_offset, angle_offset))
            random.shuffle(augmentation_params[i])
        
        # 执行增强并保存（保存路径包含instance_environment_augj）
        for j in range(augmentation_factor):
            # 构建保存目录名称：instance_name_environment_name_augj
            save_dir_name = f'{instance_name}_{environment_name}_aug{j}'
            for i in range(4):
                frame_offset, angle_offset = augmentation_params[i][j]
                # 帧截取
                aug_signal = origin_signals[i][frame_offset:frame_offset+target_frame, :]
                # 角度旋转增强
                rotation_factor = np.exp(1j * angle_offset)
                aug_signal = aug_signal * rotation_factor
                
                # 构建完整保存路径
                save_path = os.path.join(
                    data_after_augmentation, category_name, save_dir_name, "models", f"{i}.mat"
                )
                # 创建多级目录（如果不存在）
                save_parent_dir = os.path.dirname(save_path)
                if not os.path.exists(save_parent_dir):
                    os.makedirs(save_parent_dir, exist_ok=True)  # exist_ok避免目录已存在的异常
                # 保存增强后的信号
                savemat(save_path, {'data': aug_signal})
        
        print(f"Completed augmentation for: {category_name}/{instance_name}/{environment_name}")

if __name__ == "__main__":
    # 检查输入目录是否存在
    if not os.path.exists(data_after_remove_background):
        raise FileNotFoundError(f"Input directory not found: {data_after_remove_background}")
    
    # 遍历所有类别
    category_list = os.listdir(data_after_remove_background)
    for category in category_list:
        category_dir = os.path.join(data_after_remove_background, category)
        if not os.path.isdir(category_dir):
            print(f"Skipping non-category directory: {category_dir}")
            continue
        
        # 遍历类别下的所有instance
        instance_list = os.listdir(category_dir)
        for instance in instance_list:
            instance_path = os.path.join(category_dir, instance)
            if not os.path.isdir(instance_path):
                print(f"Skipping non-instance directory: {instance_path}")
                continue
            
            # 执行增强（包含environment遍历逻辑）
            try:
                augmentation(instance_path)
            except Exception as e:
                print(f"Error processing {category}/{instance}: {str(e)}")