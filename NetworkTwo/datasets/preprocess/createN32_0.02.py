import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d

# 源根目录和目标根目录（不含具体类别）
src_root = "/home/zhang_muxin/Signal2PC/datasets/data/N32"
dst_root = "/home/zhang_muxin/Signal2PC/datasets/data/N32_0.02"

# 需要处理的类别目录列表
categories = [
    "扳手", "笔记本电脑", "刀具", "键盘", 
    "手机", "水杯", "显示器", "易拉罐"
]

# 遍历每个类别目录
for category in categories:
    src_category = os.path.join(src_root, category)
    dst_category = os.path.join(dst_root, category)
    
    # 检查源类别目录是否存在
    if not os.path.exists(src_category):
        print(f"跳过不存在的目录: {src_category}")
        continue
    
    # 创建目标类别根目录
    os.makedirs(dst_category, exist_ok=True)
    print(f"正在处理类别: {category}")

    # 遍历源类别目录
    for root, _, files in os.walk(src_category):
        # 计算相对路径以保持目录结构
        rel_path = os.path.relpath(root, src_category)
        dst_dir = os.path.join(dst_category, rel_path)
        os.makedirs(dst_dir, exist_ok=True)
        
        for file in files:
            if file.endswith(".mat"):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_dir, file)
                
                try:
                    # 读取MAT文件
                    mat_data = loadmat(src_file)
                    data = mat_data["data"]
                    
                    # 验证数据形状
                    if data.shape != (2800, 9):
                        print(f"  跳过 {file}: 形状 {data.shape} 不符合要求 (2800, 9)")
                        continue
                    
                    # 1. 等间隔采样1400行
                    num_sampled = 1400
                    indices = np.linspace(0, 2800 - 1, num_sampled, dtype=int)
                    sampled_data = data[indices, :]
                    
                    # 2. 三次样条插值回2800行
                    x_old = indices
                    x_new = np.arange(2800)
                    f = interp1d(x_old, sampled_data, kind="cubic", axis=0)
                    new_data = f(x_new)
                    
                    # 【关键修改】显式转换为 complex64 (Complex Single)
                    new_data = new_data.astype(np.complex64)
                    
                    # 3. 保存到目标目录
                    savemat(dst_file, {"data": new_data})
                    print(f"  已处理: {file} (保存为 {new_data.dtype})")
                    
                except Exception as e:
                    print(f"  处理 {file} 时出错: {str(e)}")