import os
import random

def main():
    # ================= 配置参数 =================
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2_2025_11_28_scene_nomask"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2_2025_11_28_scene_all/meta"

    val_filename = "val_tf_scene_nomask_键盘.txt"
    train_filename = "train_tf_scene_nomask_键盘.txt"
    test_filename = "test_tf_scene_nomask_键盘.txt"

    fixed_prefix = "Data/ShapeNetP2M/"

    # 【新增配置】指定要划分的物体类别名称
    # 如果设置为 None 或 ""，则处理所有物体。
    # 如果设置为具体的文件夹名（如 "扳手" 或 "扳手-遮挡"），则只处理该类别。
    target_category = "键盘" 
    # ===========================================

    os.makedirs(output_meta_dir, exist_ok=True)

    train_instances = []
    val_instances = []
    test_instances = []

    if not os.path.exists(root_dir):
        print(f"错误：根目录不存在 {root_dir}")
        return

    if target_category:
        print(f"正在处理指定类别: 【{target_category}】...")
    else:
        print("正在处理所有类别...")

    # 遍历二级目录 (Category)
    category_list = os.listdir(root_dir)
    for sec_dir in category_list:
        sec_path = os.path.join(root_dir, sec_dir)
        if not os.path.isdir(sec_path):
            continue

        # =================== 新增：类别过滤逻辑 ===================
        # 如果指定了类别，且当前目录名不等于指定类别，则跳过
        if target_category and sec_dir != target_category:
            continue
        # ========================================================

        # 遍历三级目录 (Instance)
        for tert_dir in os.listdir(sec_path):
            tert_path = os.path.join(sec_path, tert_dir)
            if not os.path.isdir(tert_path):
                continue
            
            # =================== 核心划分逻辑 ===================
            # 兼容 "1_0", "1_0_aug" 等格式
            # 提取第一个下划线前的数字
            
            parts = tert_dir.split('_')
            
            # 提取第一部分作为 batch_id
            if not parts or not parts[0].isdigit():
                continue
            
            batch_id = parts[0] 
            instance_info = (sec_dir, tert_dir)

            # 根据 batch_id 分配到对应集合
            # 1-4 -> Train, 5 -> Val, 6 -> Test
            
            if batch_id == "6":
                test_instances.append(instance_info)
            
            elif batch_id == "5":
                val_instances.append(instance_info)
            
            elif batch_id in ["1", "2", "3", "4"]:
                train_instances.append(instance_info)
            
            else:
                pass
            # ===================================================

    # 检查是否找到了数据
    total_found = len(train_instances) + len(val_instances) + len(test_instances)
    if total_found == 0:
        print(f"警告：未找到类别为 '{target_category}' 的符合要求的数据。请检查文件夹名称是否完全匹配。")
        return

    # ---------------------------------------------
    # 第二步：打乱列表顺序 (Shuffle)
    # ---------------------------------------------
    random.shuffle(train_instances)
    random.shuffle(val_instances)
    random.shuffle(test_instances)

    print(f"\n--- 实例划分统计 (Group Level) ---")
    print(f"目标类别: {target_category if target_category else 'ALL'}")
    print(f"Train (Batch 1-4): {len(train_instances)}")
    print(f"Val   (Batch 5)  : {len(val_instances)}")
    print(f"Test  (Batch 6)  : {len(test_instances)}")
    print(f"----------------------------------")

    # ---------------------------------------------
    # 第三步：展开并写入文件
    # ---------------------------------------------
    def write_groups_to_file(instance_list, filename):
        full_path = os.path.join(output_meta_dir, filename)
        line_count = 0
        
        with open(full_path, 'w', encoding='utf-8') as f:
            for sec_dir, tert_dir in instance_list:
                base_path = os.path.join(root_dir, sec_dir, tert_dir)
                
                # 遍历 0-3 四个视角
                for i in range(4): 
                    # 检查文件是否存在
                    # 根据之前的约定，物理文件可能在 tert_dir 下，也可能在 models 下
                    # 这里默认检查直接在 tert_dir 下的情况
                    mat_file = os.path.join(base_path, f"{i}.mat")
                    
                    # 如果物理文件在 models 文件夹里，请取消下行注释：
                    # mat_file = os.path.join(base_path, "models", f"{i}.mat")

                    if os.path.exists(mat_file):
                        # 写入路径不包含 models
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