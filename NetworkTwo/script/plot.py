import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
root = '/home/zhang_muxin'

# 定义两个root目录
simulate_root = 'shapenetS2M_simulate_cut'
simulate_root = os.path.join(root, simulate_root)
gt_root = 'shapenetS2M_GT_signal'
gt_root = os.path.join(root, gt_root)
output_dir = './output'
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取物品种类和物品名称
categories = os.listdir(simulate_root)
for category in categories:
    category_path_simulate = os.path.join(simulate_root, category)
    category_path_gt = os.path.join(gt_root, category)
    
    items = os.listdir(category_path_simulate)
    for item in items:
        item_path_simulate = os.path.join(category_path_simulate, item, 'models')
        item_path_gt = os.path.join(category_path_gt, item, 'models')
        
        # 创建一个新的figure
        plt.figure(figsize=(12, 8))
        
        # 遍历四个视角
        for i in range(4):
            # 加载模拟信号
            simulate_file = os.path.join(item_path_simulate, f'{i}.mat')
            simulate_data = loadmat(simulate_file)['data']
            
            # 加载真实信号
            gt_file = os.path.join(item_path_gt, f'{i}.mat')
            gt_data = loadmat(gt_file)['data']
            
            # 计算信号的幅值
            simulate_magnitude = np.abs(simulate_data)
            gt_magnitude = np.abs(gt_data)
            
            # 绘制模拟信号的热度图
            plt.subplot(2, 4, i+1)
            plt.imshow(simulate_magnitude, aspect='auto', cmap='viridis')
            plt.title(f'Simulated Signal (View {i})')
            plt.xlabel('Channel')
            plt.ylabel('Sample Index')
            plt.colorbar(label='Magnitude')
            
            # 绘制真实信号的热度图
            plt.subplot(2, 4, i+5)
            plt.imshow(gt_magnitude, aspect='auto', cmap='viridis')
            plt.title(f'GT Signal (View {i})')
            plt.xlabel('Channel')
            plt.ylabel('Sample Index')
            plt.colorbar(label='Magnitude')
        
        # 保存图像
        output_path = os.path.join(output_dir, f'{category}_{item}.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

print("所有图像已保存到output目录中。")