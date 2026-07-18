import os
import random
# 删除了未使用的 shutil 库

def main():
    # 配置参数
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/N32"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/N32/meta"
    val_filename = "val_tf_all_S2M_4views_pro_N32dataset_yilaguan.txt"
    # train_filename 已删除（不需要训练集）
    fixed_prefix = "Data/ShapeNetP2M/"
    # 新增：指定只处理的二级目录名称（请确保和你的文件夹名完全一致！）
    target_secondary_dir = "易拉罐"  # 替换为你要处理的类别名称，例如 "扳手"、"显示器" 等

    # 确保输出目录存在
    os.makedirs(output_meta_dir, exist_ok=True)

    # 收集所有三级目录
    all_tertiary_dirs = []

    # 遍历二级目录
    for secondary_dir in os.listdir(root_dir):
        # 核心修改：只处理指定的目录，其他类别直接跳过
        if secondary_dir != target_secondary_dir:
            continue
            
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

    # 核心修改：所有数据全部作为验证集，不划分训练集
    total = len(all_tertiary_dirs)
    val_count = total  # 验证集 = 全部数据
    train_count = 0    # 训练集数量为0
    val_dirs = all_tertiary_dirs  # 全部数据赋值给验证集

    # 生成文件路径并写入文件
    def write_paths(directories, output_file):
        with open(os.path.join(output_meta_dir, output_file), 'w', encoding='utf-8') as f:
            for secondary_dir, tertiary_dir in directories:
                models_path = os.path.join(root_dir, secondary_dir, tertiary_dir, "models")

                # 检查4个mat文件是否存在
                for i in range(4):
                    mat_file = os.path.join(models_path, f"{i}.mat")
                    if os.path.exists(mat_file):
                        target_path = f"{fixed_prefix}{secondary_dir}/{tertiary_dir}/models/{i}.dat"
                        f.write(target_path + '\n')

    # 核心修改：仅写入验证集文件，删除训练集写入逻辑
    write_paths(val_dirs, val_filename)

    # 简化打印信息（仅保留有效内容）
    print(f"划分完成:")
    print(f"【扳手】类别总样本数:{total}")
    print(f"Val样本数:{val_count}")
    print(f"文件已保存至:{output_meta_dir}")

if __name__ == "__main__":
    main()