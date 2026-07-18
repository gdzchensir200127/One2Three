import os
import random

def main():
    # ================= 配置参数 =================
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/scene_nomask_select"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2_2025_11_28_scene_all/meta"

    val_filename = "val_tf_scene_nomask_baseline.txt"
    train_filename = "train_tf_scene_nomask_baseline.txt"
    test_filename = "test_tf_scene_nomask_baseline.txt"

    fixed_prefix = "Data/ShapeNetP2M/"
    # ===========================================

    os.makedirs(output_meta_dir, exist_ok=True)

    train_instances = []
    val_instances = []
    test_instances = []

    if not os.path.exists(root_dir):
        print(f"错误：根目录不存在 {root_dir}")
        return

    print("正在遍历目录并根据格式 (e.g., 1_0, 2_1) 进行划分...")

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
            
            # =================== 核心划分逻辑 ===================
            # 命名格式示例: "1_0", "1_1", "2_0", "6_2"
            # 也就是: {BatchID}_{SubID}
            
            parts = tert_dir.split('_')
            
            # 提取第一个下划线前的部分作为 batch_id
            if not parts or not parts[0].isdigit():
                # 跳过不符合规则的文件夹
                continue
            
            batch_id = parts[0] 
            instance_info = (sec_dir, tert_dir)

            # 根据 batch_id 分配到对应集合
            # 逻辑：
            # 6 开头 -> Test
            # 5 开头 -> Val
            # 1, 2, 3, 4 开头 -> Train
            
            if batch_id == "6":
                test_instances.append(instance_info)
            
            elif batch_id == "5":
                val_instances.append(instance_info)
            
            elif batch_id in ["1", "2", "3", "4"]:
                train_instances.append(instance_info)
            
            else:
                # 处理其他可能的数字（如 0 或 >6），默认忽略
                pass
            # ===================================================

    # ---------------------------------------------
    # 第二步：打乱列表顺序 (Shuffle)
    # ---------------------------------------------
    random.shuffle(train_instances)
    random.shuffle(val_instances)
    random.shuffle(test_instances)

    print(f"\n--- 实例划分统计 (Group Level) ---")
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
                # 基础路径
                base_path = os.path.join(root_dir, sec_dir, tert_dir)
                
                # 遍历 0-3 四个视角
                for i in range(4): 
                    # 检查文件是否存在
                    # 假设文件结构是: root/Category/1_0/0.mat
                    # 如果您的物理文件依然在 models 文件夹里 (root/Category/1_0/models/0.mat)，
                    # 请取消下面被注释掉的那行代码的注释，并注释掉当前使用的那行。
                    
                    mat_file = os.path.join(base_path, f"{i}.mat")
                    # mat_file = os.path.join(base_path, "models", f"{i}.mat") # 如果物理文件在models里用这行
                    
                    # 只要物理文件存在，就写入路径
                    if os.path.exists(mat_file):
                        # 生成写入路径：Data/ShapeNetP2M/类别/实例/i.dat
                        # 严格遵守：路径字符串中不包含 "/models/"
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