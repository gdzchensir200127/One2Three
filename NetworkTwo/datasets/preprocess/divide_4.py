import os
import random

def main():
    # ================= 配置参数 =================
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2_2025_11_20"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/dataset_2025_11_20_2aug_N_2_1/meta"

    val_filename = "val_tf_all_S2M_4views_pro_realdataset_unseen_2025_11_20_2aug_2.txt"
    train_filename = "train_tf_all_S2M_4views_pro_realdataset_unseen_2025_11_20_2aug_2.txt"
    test_filename = "test_tf_all_S2M_4views_pro_realdataset_unseen_2025_11_20_2aug_2.txt"

    fixed_prefix = "Data/ShapeNetP2M/"
    # ===========================================

    os.makedirs(output_meta_dir, exist_ok=True)

    # ---------------------------------------------
    # 第一步：以“实例（三级目录）”为单位收集数据
    # ---------------------------------------------
    # 列表里存放的是元组：(二级目录名, 三级目录名)
    # 例如：('02691156', 'instance_0')
    all_instances = []

    if not os.path.exists(root_dir):
        print(f"错误：根目录不存在 {root_dir}")
        return

    print("正在遍历目录收集实例组（每组包含4个视角）...")

    # 遍历二级目录 (Category)
    for sec_dir in os.listdir(root_dir):
        sec_path = os.path.join(root_dir, sec_dir)
        if not os.path.isdir(sec_path):
            continue

        # 遍历三级目录 (Instance) —— 这是我们划分的最小原子单位
        for tert_dir in os.listdir(sec_path):
            tert_path = os.path.join(sec_path, tert_dir)
            if not os.path.isdir(tert_path):
                continue
            
            # 作为一个整体加入列表
            all_instances.append((sec_dir, tert_dir))

    total_instances = len(all_instances)
    if total_instances == 0:
        print("未找到任何实例目录。")
        return

    print(f"共收集到 {total_instances} 个实例组。")

    # ---------------------------------------------
    # 第二步：随机打乱“实例”列表
    # ---------------------------------------------
    random.shuffle(all_instances)

    # ---------------------------------------------
    # 第三步：按 4:1:1 比例划分实例
    # ---------------------------------------------
    unit_count = total_instances // 6

    test_count = unit_count      # 1份
    val_count = unit_count       # 1份
    train_count = total_instances - val_count - test_count # 剩余4份

    # 切分列表
    test_instances = all_instances[:test_count]
    val_instances = all_instances[test_count : test_count + val_count]
    train_instances = all_instances[test_count + val_count:]

    print(f"\n--- 实例划分统计 (Group Level) ---")
    print(f"Train 实例组: {len(train_instances)}")
    print(f"Val   实例组: {len(val_instances)}")
    print(f"Test  实例组: {len(test_instances)}")
    print(f"----------------------------------")

    # ---------------------------------------------
    # 第四步：展开并写入文件
    # ---------------------------------------------
    # 辅助函数：将实例列表展开为具体的 0-3.dat 路径并写入
    def write_groups_to_file(instance_list, filename):
        full_path = os.path.join(output_meta_dir, filename)
        line_count = 0
        
        with open(full_path, 'w', encoding='utf-8') as f:
            for sec_dir, tert_dir in instance_list:
                # 这里的逻辑是：只要这个实例被分到了某个集，
                # 它下面的 0, 1, 2, 3 四个视角全部紧挨着写入该集。
                
                # 获取实际物理路径用于检查文件是否存在（可选，防止写入空路径）
                base_path = os.path.join(root_dir, sec_dir, tert_dir)
                
                for i in range(4): # 0, 1, 2, 3
                    # 检查 mat 文件是否存在，确保数据有效
                    mat_file = os.path.join(base_path, f"{i}.mat")
                    if os.path.exists(mat_file):
                        # 写入格式：Data/ShapeNetP2M/category/instance/i.dat
                        line = f"{fixed_prefix}{sec_dir}/{tert_dir}/{i}.dat"
                        f.write(line + '\n')
                        line_count += 1
        return line_count

    # 执行写入
    n_train = write_groups_to_file(train_instances, train_filename)
    n_val = write_groups_to_file(val_instances, val_filename)
    n_test = write_groups_to_file(test_instances, test_filename)

    print(f"\n--- 最终写入行数统计 (Line Level) ---")
    print(f"Train 总行数: {n_train}")
    print(f"Val   总行数: {n_val}")
    print(f"Test  总行数: {n_test}")
    print(f"元数据文件已保存至: {output_meta_dir}")

if __name__ == "__main__":
    main()