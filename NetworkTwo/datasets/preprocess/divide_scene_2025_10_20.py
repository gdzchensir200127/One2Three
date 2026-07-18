'''import os
import random
import shutil


def main():
    # 配置参数
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2025_10_19_func"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2025_10_19_func"
    val_filename = "val_tf_all_S2M_4views_pro_realdataset_unseen_2025_10_20_scene.txt"
    train_filename = "train_tf_all_S2M_4views_pro_realdataset_unseen_2025_10_20_scene.txt"
    fixed_prefix = "Data/ShapeNetP2M/"

    # 确保输出目录存在
    os.makedirs(output_meta_dir, exist_ok=True)

    
    train_dirs = []
    val_dirs = []

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
                
                # 检查是否至少有两部分（我们需要 parts[1]）
                if len(parts) >= 2:
                    scene_id_str = parts[1] # 获取第二个数字 'Y'

                    # 根据 Y 的值分配
                    if scene_id_str == '0' or scene_id_str == '1':
                        train_dirs.append((secondary_dir, tertiary_dir))
                    elif scene_id_str == '2':
                        val_dirs.append((secondary_dir, tertiary_dir))
                    else:
                        print(f"警告：跳过目录 {tertiary_path}，未识别的场景ID: '{scene_id_str}'")
                else:
                    print(f"警告：跳过目录 {tertiary_path}，命名不符合 'X_Y_...' 格式。")

            except Exception as e:
                print(f"错误：处理目录 {tertiary_path} 时出错: {e}")
            

    # total = len(all_tertiary_dirs)
    # val_count = total // 3
    # train_count = total - val_count
    # random.shuffle(all_tertiary_dirs)
    # val_dirs = all_tertiary_dirs[:val_count]
    # train_dirs = all_tertiary_dirs[val_count:]
    

    # 重新计算数量
    train_count = len(train_dirs)
    val_count = len(val_dirs)
    total = train_count + val_count

    # 生成文件路径并写入文件 (这部分函数保持不变)
    def write_paths(directories, output_file):
        with open(os.path.join(output_meta_dir, output_file), 'w', encoding='utf-8') as f:
            for secondary_dir, tertiary_dir in directories:
                
                
                # 构建原始路径 (现在直接指向三级目录)
                mat_files_path = os.path.join(root_dir, secondary_dir, tertiary_dir)

                # 检查4个mat文件是否存在
                for i in range(4):
                    # 直接在 mat_files_path 下查找 .mat 文件
                    mat_file = os.path.join(mat_files_path, f"{i}.mat")
                    if os.path.exists(mat_file):
                        # 构建目标路径格式 (也不再包含 "models")
                        target_path = f"{fixed_prefix}{secondary_dir}/{tertiary_dir}/{i}.dat"
                        f.write(target_path + '\n')
                # ---------------------------------------------

    # 写入val和train文件
    write_paths(val_dirs, val_filename)
    write_paths(train_dirs, train_filename)

    print(f"划分完成：")
    print(f"总三级目录数：{total}")
    # 更新打印信息以反映新规则
    print(f"Val目录数 (场景 2)：{val_count}")
    print(f"Train目录数 (场景 0 和 1)：{train_count}")
    print(f"文件已保存至：{output_meta_dir}")


if __name__ == "__main__":
    main()'''
import os
import random
import shutil


def main():
    # 配置参数
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_VAS_func"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_VAS_2025_10_23_T_3"
    val_filename = "val_tf_all_S2M_4views_pro_realdataset_unseen_aug_VAS_2025_10_23_T_3.txt"
    train_filename = "train_tf_all_S2M_4views_pro_realdataset_unseen_aug_VAS_2025_10_23_T_3.txt"
    fixed_prefix = "Data/ShapeNetP2M/"

    # 确保输出目录存在
    os.makedirs(output_meta_dir, exist_ok=True)

    train_dirs = []
    val_dirs = []

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

                    if scene_id_str == '0' or scene_id_str == '1':
                        train_dirs.append((secondary_dir, tertiary_dir))
                    elif scene_id_str == '2':
                        val_dirs.append((secondary_dir, tertiary_dir))
                    else:
                        print(f"警告：跳过目录 {tertiary_path}，未识别的场景ID: '{scene_id_str}'")
                else:
                    print(f"警告：跳过目录 {tertiary_path}，命名不符合 'X_Y_...' 格式。")

            except Exception as e:
                print(f"错误：处理目录 {tertiary_path} 时出错: {e}")

    # 在划分完成后，打乱训练集和验证集各自的内部顺序
    # 这不会改变“哪个文件属于训练集”，但会打乱文件列表的顺序
    # 从而解决多进程数据加载时的 I/O 瓶颈
    random.shuffle(train_dirs)
    random.shuffle(val_dirs)
    # ------------------- 修改结束 -------------------

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

    print(f"划分完成：")
    print(f"总三级目录数：{total}")
    print(f"Val目录数 (场景 2)：{val_count}")
    print(f"Train目录数 (场景 0 和 1)：{train_count}")
    print(f"文件已保存至：{output_meta_dir}")
    print("已添加 random.shuffle() 来优化 I/O 性能。")


if __name__ == "__main__":
    main()    