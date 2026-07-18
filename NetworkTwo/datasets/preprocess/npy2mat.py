import os
import numpy as np
from scipy.io import savemat

# 定义各个目录路径
real_dir = '/home/zhang_muxin/Signal2PC/datasets/data/DATASET_fornet/Simulate_nomask/signal_norm_single_real/'
imag_dir = '/home/zhang_muxin/Signal2PC/datasets/data/DATASET_fornet/Simulate_nomask/signal_norm_single_imag/'
output_dir = '/home/zhang_muxin/Signal2PC/datasets/data/DATASET_fornet/Simulate_nomask/signal_norm_single/'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 获取实部目录下所有的npy文件
real_files = [f for f in os.listdir(real_dir) if f.endswith('.npy')]

print(f"找到 {len(real_files)} 个实部文件，开始合并处理...")

# 遍历每个实部文件，匹配对应的虚部文件并合并
for real_file in real_files:
    # 获取文件名（不含后缀）
    file_name = os.path.splitext(real_file)[0]
    
    # 构建对应的虚部文件路径和输出mat文件路径
    imag_file = os.path.join(imag_dir, f"{file_name}.npy")
    output_mat = os.path.join(output_dir, f"{file_name}.mat")
    
    # 检查虚部文件是否存在
    if not os.path.exists(imag_file):
        print(f"警告：{imag_file} 不存在，跳过该文件")
        continue
    
    try:
        # 加载实部和虚部数据
        real_data = np.load(real_dir + real_file)
        imag_data = np.load(imag_file)
        
        # 检查实部和虚部数组维度是否一致
        if real_data.shape != imag_data.shape:
            print(f"警告：{file_name} 的实部和虚部数组维度不匹配（实部：{real_data.shape}，虚部：{imag_data.shape}），跳过该文件")
            continue
        
        # 合并为复数数组（实部 + 虚部 * 1j）
        complex_data = real_data + imag_data * 1j
        
        # 保存为mat文件（变量名设为'data'，可根据需要修改）
        savemat(output_mat, {'data': complex_data})
        
        print(f"成功处理：{file_name} -> 保存到 {output_mat}")
    
    except Exception as e:
        print(f"错误：处理 {file_name} 时发生异常 - {str(e)}")

print("\n所有文件处理完成！")