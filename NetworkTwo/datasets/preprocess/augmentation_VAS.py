import os
import numpy as np
import random
from scipy.io import *
root_dir = "/home/zhang_muxin/data/"
data_after_remove_background = os.path.join(root_dir, "remove_background")
data_after_augmentation = os.path.join(root_dir, "augmentation_VAS")
target_frame = 2800
min_frame_offset = 0
max_frame_offset = 2880-2800
min_angle_offset = 0
max_angle_offset = 2*np.pi
augmentation_factor_1 = 10
augmentation_factor = 10
def augmentation(instance_path):
    instance_name = os.path.basename(instance_path)
    category_name = os.path.basename(os.path.dirname(instance_path))

    origin_signals = []
    for i in range(4):
        signal = loadmat(os.path.join(instance_path, f"{i}.mat"))['data']
        origin_signals.append(signal)
    augmentation_params = [[] for i in range(4)]
    for i in range(4):
        for aug_1 in range(augmentation_factor_1):
            frame_offset = random.randint(min_frame_offset, max_frame_offset)
            augmentation_params[i].append(frame_offset)
    for i in range(4):
        random.shuffle(augmentation_params[i])
    angle_offsets = [random.uniform(min_angle_offset, max_angle_offset) for _ in range(4)]
    for j in range(augmentation_factor):
        for i in range(4):
            frame_offset = augmentation_params[i][j]
            aug_signal = origin_signals[i][frame_offset:frame_offset+target_frame,:]
            rotation_factor = np.exp(1j * angle_offsets[i])
            aug_signal = aug_signal * rotation_factor
            save_path = os.path.join(data_after_augmentation, category_name, f'{instance_name}_aug{j}','models', f'{i}.mat')
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            savemat(save_path, {'data':aug_signal})

if __name__ == "__main__":
    category_list = os.listdir(data_after_remove_background)
    for category in category_list:
        try:
            category_dir = os.path.join(data_after_remove_background, category)
            instance_list = os.listdir(category_dir)
            for instance in instance_list:
                instance_path = os.path.join(category_dir, instance)
                augmentation(instance_path)
        except Exception as e:
            print(e)