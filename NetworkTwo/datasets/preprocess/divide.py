import os
import random
import shutil


def main():
    # 配置参数
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_VAS_func"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_VAS_2025_10_25_T_3_1"
    val_filename = "val_tf_all_S2M_4views_pro_realdataset_unseen_aug_VAS_2025_10_25_3_1.txt"
    train_filename = "train_tf_all_S2M_4views_pro_realdataset_unseen_aug_VAS_2025_10_25_3_1.txt"
    fixed_prefix = "Data/ShapeNetP2M/"

    # 确保输出目录存在
    os.makedirs(output_meta_dir, exist_ok=True)

    # 收集所有三级目录
    all_tertiary_dirs = []

    # 遍历二级目录（如扳手，显示器等）
    for secondary_dir in os.listdir(root_dir):
        secondary_path = os.path.join(root_dir, secondary_dir)
        if not os.path.isdir(secondary_path):
            continue

        # 遍历三级目录（如7_3, 16_2等）
        for tertiary_dir in os.listdir(secondary_path):
            tertiary_path = os.path.join(secondary_path, tertiary_dir)
            if not os.path.isdir(tertiary_path):
                continue

           
            # 只要 tertiary_path 是一个目录，就认为它有效
            all_tertiary_dirs.append((secondary_dir, tertiary_dir))
            # ---------------------------------------------

    # 计算划分数量
    total = len(all_tertiary_dirs)
    val_count = total // 5
    train_count = total - val_count

    # 随机打乱并划分
    random.shuffle(all_tertiary_dirs)
    val_dirs = all_tertiary_dirs[:val_count]
    train_dirs = all_tertiary_dirs[val_count:]

    # 生成文件路径并写入文件
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
    print(f"Val目录数：{val_count}")
    print(f"Train目录数：{train_count}")
    print(f"文件已保存至：{output_meta_dir}")


if __name__ == "__main__":
    main()