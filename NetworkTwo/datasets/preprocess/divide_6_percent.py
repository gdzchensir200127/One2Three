import os
import random

def main():
    # ================= 配置参数 (可直接调整) =================
    
    # [核心修改] 数据保留比例 (0.0 - 1.0)
    # 0.7 代表保留 70% 的数据，0.5 代表 50%
    DATA_KEEP_RATIO = 0.05 
    
    # [核心修改] 随机种子 (保证每次实验选取的"70%"是固定的，方便复现)
    RANDOM_SEED = 2025 

    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2_2025_11_28_scene_nomask"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2_2025_11_28_scene_all/meta"

    # 自动根据比例修改文件名，方便区分
    # 例如: train_tf_simulate_all_vas_0.7.txt
    suffix = f"_{DATA_KEEP_RATIO}" if DATA_KEEP_RATIO < 1.0 else ""
    val_filename = f"val_tf_scene_nomask{suffix}.txt"
    train_filename = f"train_tf_scene_nomask{suffix}.txt"
    test_filename = f"test_tf_scene_nomask{suffix}.txt"

    fixed_prefix = "Data/ShapeNetP2M/"
    # ========================================================

    # 设置随机种子
    random.seed(RANDOM_SEED)

    os.makedirs(output_meta_dir, exist_ok=True)

    if not os.path.exists(root_dir):
        print(f"错误：根目录不存在 {root_dir}")
        return

    print(f"当前设置: 保留比例 = {DATA_KEEP_RATIO * 100}%")
    print(f"当前设置: 随机种子 = {RANDOM_SEED}")
    print("正在遍历目录...")

    # [结构修改] 使用字典按类别临时存储数据
    # 结构: { '类别名': { 'train': [], 'val': [], 'test': [] } }
    category_storage = {}

    # 1. 遍历并按类别分类
    category_list = sorted(os.listdir(root_dir)) # sorted保证跨平台顺序一致
    
    for sec_dir in category_list: # sec_dir 是类别名 (Category)
        sec_path = os.path.join(root_dir, sec_dir)
        if not os.path.isdir(sec_path):
            continue
        
        # 初始化该类别的数据容器
        if sec_dir not in category_storage:
            category_storage[sec_dir] = {'train': [], 'val': [], 'test': []}

        # 遍历该类别下的所有实例
        instance_list = sorted(os.listdir(sec_path))
        for tert_dir in instance_list:
            tert_path = os.path.join(sec_path, tert_dir)
            if not os.path.isdir(tert_path):
                continue
            
            # --- 解析 Batch ID ---
            parts = tert_dir.split('_')
            if not parts or not parts[0].isdigit():
                continue
            
            batch_id = parts[0] 
            instance_info = (sec_dir, tert_dir)

            # 分配到该类别下的对应集合
            if batch_id == "6":
                category_storage[sec_dir]['test'].append(instance_info)
            elif batch_id == "5":
                category_storage[sec_dir]['val'].append(instance_info)
            elif batch_id in ["1", "2", "3", "4"]:
                category_storage[sec_dir]['train'].append(instance_info)

    # 2. [核心修改] 按类别进行比例截断 (Stratified Sampling)
    final_train_instances = []
    final_val_instances = []
    final_test_instances = []

    print("\n正在按类别进行抽样...")
    
    for cat_name, datasets in category_storage.items():
        # 对该类别下的 Train, Val, Test 分别处理
        for split_name, instance_list in datasets.items():
            
            # 1. 打乱顺序
            random.shuffle(instance_list)
            
            # 2. 计算保留数量
            total_count = len(instance_list)
            keep_count = int(total_count * DATA_KEEP_RATIO)
            
            # 边界处理：如果算出来是0但原数据不为0，至少保留1个？
            # 这里保持严格比例，如果很少可能变0。如果需要至少保留1个，取消下面注释
            # if total_count > 0 and keep_count == 0: keep_count = 1
            
            # 3. 截断列表
            selected_instances = instance_list[:keep_count]
            
            # 4. 加入最终的大池子
            if split_name == 'train':
                final_train_instances.extend(selected_instances)
            elif split_name == 'val':
                final_val_instances.extend(selected_instances)
            elif split_name == 'test':
                final_test_instances.extend(selected_instances)

    # 3. 再次打乱最终的大列表 (混淆类别顺序)
    random.shuffle(final_train_instances)
    random.shuffle(final_val_instances)
    random.shuffle(final_test_instances)

    print(f"\n--- 实例划分统计 (最终保留数量) ---")
    print(f"Train : {len(final_train_instances)}")
    print(f"Val   : {len(final_val_instances)}")
    print(f"Test  : {len(final_test_instances)}")
    print(f"----------------------------------")

    # 4. 写入文件 (逻辑不变)
    def write_groups_to_file(instance_list, filename):
        full_path = os.path.join(output_meta_dir, filename)
        line_count = 0
        
        with open(full_path, 'w', encoding='utf-8') as f:
            for sec_dir, tert_dir in instance_list:
                base_path = os.path.join(root_dir, sec_dir, tert_dir)
                
                for i in range(4): 
                    # 检查路径：根据实际情况调整是否需要 "models"
                    # mat_file = os.path.join(base_path, "models", f"{i}.mat")
                    mat_file = os.path.join(base_path, f"{i}.mat")
                    
                    if os.path.exists(mat_file):
                        line = f"{fixed_prefix}{sec_dir}/{tert_dir}/{i}.dat"
                        f.write(line + '\n')
                        line_count += 1
        return line_count

    n_train = write_groups_to_file(final_train_instances, train_filename)
    n_val = write_groups_to_file(final_val_instances, val_filename)
    n_test = write_groups_to_file(final_test_instances, test_filename)

    print(f"\n--- 最终写入行数统计 (Line Level) ---")
    print(f"Train 文件 ({train_filename}): {n_train}")
    print(f"Val   文件 ({val_filename}): {n_val}")
    print(f"Test  文件 ({test_filename}): {n_test}")
    print(f"元数据文件已保存至: {output_meta_dir}")

if __name__ == "__main__":
    main()