import os
import random
import shutil


def main():
    # 配置参数
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2025_11_29_scene_nomask_2"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2025_11_29_scene_nomask_2"
    val_filename = "test_tf_scene_nomask_d1_2.txt"
    train_filename = "train_tf_scene_nomask_d1_2.txt"
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
        print(f"警告：检测到 {len(all_objects)} 个物体目录，但方案一期望 8 个。")
        print("将按检测到的物体列表继续划分...")
        
    # 训练集物体 (6个)
    O_train_S1_S2 = set(all_objects[:6]) # O1...O6
    # 测试集物体 (2个)
    O_test_S1_S2 = set(all_objects[6:])  # O7, O8

    # 训练集物体 (4个)
    O_train_S3 = set(all_objects[:4]) # O1...O4
    # 测试集物体 (4个)
    O_test_S3 = set(all_objects[4:])  # O5...O8

    print("方案一：物体分组（基于目录名排序）：")
    print(f"S1/S2 训练物体 (O1-O6): {sorted(list(O_train_S1_S2))}")
    print(f"S1/S2 测试物体 (O7-O8): {sorted(list(O_test_S1_S2))}")
    print(f"S3 训练物体 (O1-O4): {sorted(list(O_train_S3))}")
    print(f"S3 测试物体 (O5-O8): {sorted(list(O_test_S3))}")
    print("-" * 30)
    ############# 修改结束 #############


    # 遍历二级目录（如扳手，显示器等）
    for secondary_dir in os.listdir(root_dir):
        secondary_path = os.path.join(root_dir, secondary_dir)
        if not os.path.isdir(secondary_path):
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

                    
                    
                    if scene_id_str == '0' or scene_id_str == '1': # 场景 S1 或 S2
                        if secondary_dir in O_train_S1_S2:
                            train_dirs.append((secondary_dir, tertiary_dir))
                        elif secondary_dir in O_test_S1_S2:
                            val_dirs.append((secondary_dir, tertiary_dir))
                        # else:
                        #     # 物体名不在预期列表内（例如 .DS_Store），secondary_dir 检查已处理
                        #     pass

                    elif scene_id_str == '2': # 场景 S3
                        if secondary_dir in O_train_S3:
                            train_dirs.append((secondary_dir, tertiary_dir))
                        elif secondary_dir in O_test_S3:
                            val_dirs.append((secondary_dir, tertiary_dir))
                        # else:
                        #     pass
                    
                    else:
                        print(f"警告：跳过目录 {tertiary_path}，未识别的场景ID: '{scene_id_str}'")
                    
                   
                
                else:
                    print(f"警告：跳过目录 {tertiary_path}，命名不符合 'X_Y_...' 格式。")

            except Exception as e:
                print(f"错误：处理目录 {tertiary_path} 时出错: {e}")

  
    random.shuffle(train_dirs)
    random.shuffle(val_dirs)
    
    # 重新计算数量
    train_count = len(train_dirs)
    val_count = len(val_dirs)
    total = train_count + val_count

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

    print(f"划分完成（方案一：未见组合）：")
    print(f"总三级目录数：{total}")  
    print(f"Val (Test) 目录数 (S1/S2的O7-O8, S3的O5-O8)：{val_count}")
    print(f"Train 目录数 (S1/S2的O1-O6, S3的O1-O4)：{train_count}")
    print(f"比例 (Train/Val)：{train_count / val_count if val_count > 0 else 'N/A'} : 1")  
    print(f"文件已保存至：{output_meta_dir}")
    print("已添加 random.shuffle() 来优化 I/O 性能。")


if __name__ == "__main__":
    main()