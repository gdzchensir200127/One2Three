import os
import shutil
from pathlib import Path

# 配置路径
source_dir = "/home/zhang_muxin/Signal2PC/datasets/data/fornet/Simulate_mask/"
base_target_dir = "/home/zhang_muxin/Signal2PC/datasets/data/fornet/Simulate_mask/neworg/"

# 检查源目录是否存在
if not os.path.exists(source_dir):
    print(f"错误：源目录不存在 - {source_dir}")
    exit(1)

# 遍历源目录下所有.mat文件
for filename in os.listdir(source_dir):
    if filename.endswith(".mat") and "_" in filename:
        # 分割文件名（按"_"分割为四部分）
        parts = filename.split("_")
        if len(parts) != 4:
            print(f"警告：文件名格式不正确，跳过 - {filename}")
            continue
        
        # 解析各部分
        part1 = parts[0]          # 第一部分（如0、4）
        part2 = parts[1]          # 第二部分（如扳手）
        part3 = parts[2]          # 第三部分（如1、5）
        part4 = parts[3].replace(".mat", "")  # 第四部分（去掉.mat后缀，如0、3）
        
        # 构建目标目录路径：base_target_dir/part2/part3/part1/models
        target_dir = os.path.join(base_target_dir, part2, part3, part1, "models")
        
        # 构建目标文件路径
        target_filename = f"{part4}.mat"
        target_filepath = os.path.join(target_dir, target_filename)
        
        # 构建源文件路径
        source_filepath = os.path.join(source_dir, filename)
        
        try:
            # 创建目标目录（递归创建不存在的目录）
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            
            # 复制文件（如果目标文件已存在，会覆盖，如需跳过可添加判断）
            shutil.copy2(source_filepath, target_filepath)  # copy2保留文件元数据
            print(f"成功处理：{filename} → {target_filepath}")
        except Exception as e:
            print(f"处理失败：{filename} - 错误信息：{str(e)}")

print("\n所有文件处理完成！")