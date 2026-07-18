import os
import shutil


def main():
    # 配置参数
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/unseen_scene"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data"
    val_filename = "val_tf_all_S2M_4views_pro_unseen_scene_dataset.txt"  # 对应test目录
    train_filename = "train_tf_all_S2M_4views_pro_unseen_scene_dataset.txt"  # 对应train目录
    fixed_prefix = "Data/ShapeNetP2M/"

    # 定义train和test的父目录（固定目录结构）
    train_parent_dir = os.path.join(root_dir, "train")
    val_parent_dir = os.path.join(root_dir, "test")

    # 确保输出目录存在
    os.makedirs(output_meta_dir, exist_ok=True)

    # 辅助函数：收集指定父目录下的所有有效(二级目录, 三级目录)对
    def collect_tertiary_dirs(parent_dir):
        tertiary_dirs = []
        # 检查父目录是否存在
        if not os.path.isdir(parent_dir):
            print(f"警告：目录 {parent_dir} 不存在，请检查路径配置")
            return tertiary_dirs
        
        # 遍历二级目录（如刀具、扳手等）
        for secondary_dir in os.listdir(parent_dir):
            secondary_path = os.path.join(parent_dir, secondary_dir)
            if not os.path.isdir(secondary_path):
                continue  # 跳过非目录文件
            
            # 遍历三级目录（如6_15_nomask_aug99等）
            for tertiary_dir in os.listdir(secondary_path):
                tertiary_path = os.path.join(secondary_path, tertiary_dir)
                if not os.path.isdir(tertiary_path):
                    continue  # 跳过非目录文件
                
                # 检查是否包含models目录（有效数据标志）
                models_dir = os.path.join(tertiary_path, "models")
                if os.path.isdir(models_dir):
                    tertiary_dirs.append((secondary_dir, tertiary_dir))
        
        return tertiary_dirs

    # 收集train和test目录下的所有有效三级目录
    train_dirs = collect_tertiary_dirs(train_parent_dir)
    val_dirs = collect_tertiary_dirs(val_parent_dir)

    # 生成文件路径并写入目标文件
    def write_paths(parent_dir, directories, output_file):
        output_path = os.path.join(output_meta_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for secondary_dir, tertiary_dir in directories:
                # 构建源路径（包含train/test父目录）
                models_path = os.path.join(parent_dir, secondary_dir, tertiary_dir, "models")
                
                # 检查4个视角的mat文件是否存在，存在则写入对应dat路径（保持原格式）
                for i in range(4):
                    mat_file = os.path.join(models_path, f"{i}.mat")
                    if os.path.exists(mat_file):
                        # 目标路径不包含train/test层级，与原格式一致
                        target_path = f"{fixed_prefix}{secondary_dir}/{tertiary_dir}/models/{i}.dat"
                        f.write(target_path + '\n')
        print(f"已写入 {len(directories)} 个三级目录的文件路径到 {output_path}")

    # 分别写入train和val文件
    write_paths(train_parent_dir, train_dirs, train_filename)
    write_paths(val_parent_dir, val_dirs, val_filename)

    # 输出统计信息
    print("\n" + "="*50)
    print("划分完成！统计信息：")
    print(f"Train目录（{train_parent_dir}）：")
    print(f"  - 有效三级目录数：{len(train_dirs)}")
    print(f"Test目录（{val_parent_dir}）：")
    print(f"  - 有效三级目录数：{len(val_dirs)}")
    print("="*50)


if __name__ == "__main__":
    main()