import os

# 目标目录路径
target_dir = "/home/zhang_muxin/Signal2PC/datasets/data/fornet/Simulate_nomask"

def rename_mat_files(directory):
    # 检查目录是否存在
    if not os.path.isdir(directory):
        print(f"错误：目录 {directory} 不存在！")
        return
    
    # 遍历目录下所有文件
    for filename in os.listdir(directory):
        # 只处理.mat文件
        if filename.endswith(".mat"):
            # 分割文件名和扩展名
            name_without_ext, ext = os.path.splitext(filename)
            
            # 按下划线分割文件名部分
            name_parts = name_without_ext.split("_")
            
            # 检查是否至少有4个部分（如：0_扳手_1_1）且最后一部分是'1'
            if len(name_parts) >= 4 and name_parts[-1] == "4":
                # 将最后一部分改为'0'
                name_parts[-1] = "3"
                # 重新拼接文件名
                new_name_without_ext = "_".join(name_parts)
                new_filename = new_name_without_ext + ext
                
                # 构建完整路径
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                
                # 执行重命名（避免覆盖已存在的文件）
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"已重命名：{filename} -> {new_filename}")
                else:
                    print(f"跳过：{new_filename} 已存在，避免覆盖")
    
    print("\n重命名任务完成！")

if __name__ == "__main__":
    rename_mat_files(target_dir)