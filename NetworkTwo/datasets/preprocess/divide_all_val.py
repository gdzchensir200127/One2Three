import os
import random
import shutil


def main():
    # 配置参数
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/N32_0.1"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/N32_0.1/meta"
    val_filename = "val_tf_all_S2M_4views_pro_N32dataset.txt"
    # train_filename = "train_tf_all_S2M_4views_pro_N32dataset_banshou.txt"  # 不再需要训练集文件
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

            # 检查是否包含models目录
            models_dir = os.path.join(tertiary_path, "models")
            if os.path.isdir(models_dir):
                all_tertiary_dirs.append((secondary_dir, tertiary_dir))

    # 计算划分数量（全部作为Val）
    total = len(all_tertiary_dirs)
    val_count = total
    # train_count = 0  # 不再需要训练集计数

    # 随机打乱（保持数据随机性，若不需要打乱可注释掉）
    random.shuffle(all_tertiary_dirs)
    val_dirs = all_tertiary_dirs  # 全部数据划入Val
    # train_dirs = []  # 空列表，不再使用

    # 生成文件路径并写入文件
    def write_paths(directories, output_file):
        with open(os.path.join(output_meta_dir, output_file), 'w', encoding='utf-8') as f:
            for secondary_dir, tertiary_dir in directories:
                # 构建原始路径
                models_path = os.path.join(root_dir, secondary_dir, tertiary_dir, "models")

                # 检查4个mat文件是否存在
                for i in range(4):
                    mat_file = os.path.join(models_path, f"{i}.mat")
                    if os.path.exists(mat_file):
                        # 构建目标路径格式
                        target_path = f"{fixed_prefix}{secondary_dir}/{tertiary_dir}/models/{i}.dat"
                        f.write(target_path + '\n')

    # 仅写入val文件，跳过train文件
    write_paths(val_dirs, val_filename)
    # write_paths(train_dirs, train_filename)  # 注释掉训练集文件生成

    print(f"划分完成:")
    print(f"总三级目录数:{total}")
    print(f"Val目录数:{val_count}")
    # print(f"Train目录数:{train_count}")  # 不再打印训练集信息
    print(f"文件已保存至:{output_meta_dir}")


if __name__ == "__main__":
    main()