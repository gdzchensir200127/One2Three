import os
import random
from collections import defaultdict

def main():
    # ================= 配置参数 =================
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2_2025_11_20"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/dataset_2025_11_20_2aug_N_2_1/meta"

    val_filename = "val_tf_all_S2M_4views_pro_realdataset_unseen_2025_11_20_2aug_3.txt"
    train_filename = "train_tf_all_S2M_4views_pro_realdataset_unseen_2025_11_20_2aug_3.txt"
    test_filename = "test_tf_all_S2M_4views_pro_realdataset_unseen_2025_11_20_2aug_3.txt"

    fixed_prefix = "Data/ShapeNetP2M/"
    # ===========================================

    os.makedirs(output_meta_dir, exist_ok=True)

    # ---------------------------------------------
    # 第一步：按类别收集实例（Stratified Collection）
    # ---------------------------------------------
    # 字典结构： { '类别名(二级目录)': [ (二级, 三级), (二级, 三级)... ], ... }
    data_by_category = defaultdict(list)

    if not os.path.exists(root_dir):
        print(f"错误：根目录不存在 {root_dir}")
        return

    print("正在按类别遍历目录...")

    # 遍历二级目录 (Category)
    category_list = os.listdir(root_dir)
    for sec_dir in category_list:
        sec_path = os.path.join(root_dir, sec_dir)
        if not os.path.isdir(sec_path):
            continue

        # 遍历三级目录 (Instance)
        for tert_dir in os.listdir(sec_path):
            tert_path = os.path.join(sec_path, tert_dir)
            if not os.path.isdir(tert_path):
                continue
            
            # 将实例存入对应的类别列表中
            data_by_category[sec_dir].append((sec_dir, tert_dir))

    print(f"共找到 {len(data_by_category)} 个类别。")

    # ---------------------------------------------
    # 第二步：在每个类别内部进行随机划分 (Stratified Split)
    # ---------------------------------------------
    final_train_instances = []
    final_val_instances = []
    final_test_instances = []

    # 遍历每一个类别，单独处理
    for cat_name, instance_list in data_by_category.items():
        total_in_cat = len(instance_list)
        
        # 1. 内部打乱（保证随机性）
        random.shuffle(instance_list)

        # 2. 按 4:1:1 计算切分点
        unit_count = total_in_cat // 6
        test_count = unit_count
        val_count = unit_count
        # 剩余的给训练集
        train_count = total_in_cat - test_count - val_count

        # 3. 切分并加入全局列表
        # 测试集部分
        final_test_instances.extend(instance_list[:test_count])
        # 验证集部分
        final_val_instances.extend(instance_list[test_count : test_count + val_count])
        # 训练集部分
        final_train_instances.extend(instance_list[test_count + val_count:])
        
        # (可选) 打印每个类别的划分情况，方便调试
        # print(f"类别 {cat_name}: 总数{total_in_cat} -> Train:{train_count}, Val:{val_count}, Test:{test_count}")

    # ---------------------------------------------
    # 第三步：再次打乱全局列表（可选）
    # ---------------------------------------------
    # 虽然我们在类别内部打乱了，但现在的列表顺序是：类别A的所有数据 -> 类别B的所有数据...
    # 为了让 txt 文件里的数据也是混合的（非排序的），建议对最终结果再 shuffle 一次。
    random.shuffle(final_train_instances)
    random.shuffle(final_val_instances)
    random.shuffle(final_test_instances)

    print(f"\n--- 实例划分统计 (Group Level) ---")
    print(f"Train 实例组: {len(final_train_instances)}")
    print(f"Val   实例组: {len(final_val_instances)}")
    print(f"Test  实例组: {len(final_test_instances)}")
    print(f"----------------------------------")

    # ---------------------------------------------
    # 第四步：展开并写入文件
    # ---------------------------------------------
    def write_groups_to_file(instance_list, filename):
        full_path = os.path.join(output_meta_dir, filename)
        line_count = 0
        
        with open(full_path, 'w', encoding='utf-8') as f:
            for sec_dir, tert_dir in instance_list:
                base_path = os.path.join(root_dir, sec_dir, tert_dir)
                # 保持 0,1,2,3 四个视角在一起
                for i in range(4): 
                    mat_file = os.path.join(base_path, f"{i}.mat")
                    if os.path.exists(mat_file):
                        line = f"{fixed_prefix}{sec_dir}/{tert_dir}/{i}.dat"
                        f.write(line + '\n')
                        line_count += 1
        return line_count

    # 执行写入
    n_train = write_groups_to_file(final_train_instances, train_filename)
    n_val = write_groups_to_file(final_val_instances, val_filename)
    n_test = write_groups_to_file(final_test_instances, test_filename)

    print(f"\n--- 最终写入行数统计 (Line Level) ---")
    print(f"Train 总行数: {n_train}")
    print(f"Val    总行数: {n_val}")
    print(f"Test  总行数: {n_test}")
    print(f"元数据文件已保存至: {output_meta_dir}")

if __name__ == "__main__":
    main()