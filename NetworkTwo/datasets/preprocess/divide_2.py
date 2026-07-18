import os
import random
import shutil


def main():
    # 配置参数
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2025_11_29_scene_nomask_2"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aaug_2025_11_29_scene_nomask_2"

    val_filename = "test_tf_scene_nomask_d2_2.txt"
    train_filename = "train_tf_scene_nomask_d2_2.txt"
    
    fixed_prefix = "Data/ShapeNetP2M/"

    # 确保输出目录存在
    os.makedirs(output_meta_dir, exist_ok=True)

    train_dirs = []
    val_dirs = []

  
    try:
        all_objects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        all_objects.sort() # 排序以保证划分一致性
    except FileNotFoundError:
        print(f"错误：根目录 {root_dir} 未找到。")
        return

    if len(all_objects) != 8:
        print(f"警告：检测到 {len(all_objects)} 个物体目录，但方案二期望 8 个。")
        print("将按 50/50 比例切分检测到的物体列表...")
    
    # 假设 all_objects = ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8']
    split_index = len(all_objects) // 2 
    
    O_train = set(all_objects[:split_index]) # O1...O4
    O_test = set(all_objects[split_index:])  # O5...O8

    # 3. 定义场景分组
    S_train = {'0', '1'} # 场景 S1, S2
    S_test = {'2'}       # 场景 S3

    print("方案二 (严格版)：分组（基于目录名排序）：")
    print(f"训练场景 (S_train): {S_train}")
    print(f"测试场景 (S_test): {S_test}")
    print(f"训练物体 (O_train): {sorted(list(O_train))}")
    print(f"测试物体 (O_test): {sorted(list(O_test))}")
    print("-" * 30)
   


    # 遍历二级目录（物体）
    for secondary_dir in os.listdir(root_dir):
        secondary_path = os.path.join(root_dir, secondary_dir)
        if not os.path.isdir(secondary_path):
            continue
        
        # 检查这个物体是否在我们的总列表里 (防止 .DS_Store 等)
        if secondary_dir not in all_objects:
            continue

        # 遍历三级目录（如1_0_aug0, 1_2_aug0等）
        for tertiary_dir in os.listdir(secondary_path):
            tertiary_path = os.path.join(secondary_path, tertiary_dir)
            if not os.path.isdir(tertiary_path):
                continue
            
            try:
                parts = tertiary_dir.split('_')
                
                if len(parts) >= 2:
                    scene_id_str = parts[1] 
                    object_name = secondary_dir # 物体名就是二级目录名

                   
                    
                    if (scene_id_str in S_train) and (object_name in O_train):
                        # 属于 (S1/S2) x (O1-O4) -> 加入训练集
                        train_dirs.append((secondary_dir, tertiary_dir))
                        
                    elif (scene_id_str in S_test) and (object_name in O_test):
                        # 属于 (S3) x (O5-O8) -> 加入测试集
                        val_dirs.append((secondary_dir, tertiary_dir))
                        
                    else:
                       
                        # (S1/S2) x (O5-O8) or (S3) x (O1-O4)
                        pass
                    
                   
                
                else:
                    print(f"警告：跳过目录 {tertiary_path}，命名不符合 'X_Y_...' 格式。")

            except Exception as e:
                print(f"错误：处理目录 {tertiary_path} 时出错: {e}")

    # 在划分完成后，打乱训练集和验证集各自的内部顺序
    random.shuffle(train_dirs)
    random.shuffle(val_dirs)
   

    # 重新计算数量
    train_count = len(train_dirs)
    val_count = len(val_dirs)
    total_used = train_count + val_count 

    # 生成文件路径并写入文件 (这部分函数保持不变)
    def write_paths(directories, output_file):
        with open(os.path.join(output_meta_dir, output_file), 'w', encoding='utf-8') as f:
            for secondary_dir, tertiary_dir in directories:
                
                mat_files_path = os.path.join(root_dir, secondary_dir, tertiary_dir)

                for i in range(4):
                    mat_file = os.path.join(mat_files_path, f"{i}.mat")
                    if os.path.exists(mat_file):
                        target_path = f"{fixed_prefix}{secondary_dir}/{tertiary_dir}/{i}.dat"
                        f.write(target_path + '\n')

    # 写入val和train文件
    write_paths(val_dirs, val_filename)
    write_paths(train_dirs, train_filename)

    print(f"划分完成（方案二：未见场景+未见物体）：")
    print(f"总共使用的三级目录数：{total_used}")
    print(f"Val (Test) 目录数 (S3 x O5-O8)：{val_count}")
    print(f"Train 目录数 (S1/S2 x O1-O4)：{train_count}")
    print(f"比例 (Train/Val)：{train_count / val_count if val_count > 0 else 'N/A'} : 1")
    print(f"文件已保存至：{output_meta_dir}")
    print("已添加 random.shuffle() 来优化 I/O 性能。")


if __name__ == "__main__":
    main()