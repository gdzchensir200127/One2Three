import os
import random
import shutil

def main():
    # --- 1. 基础配置 ---
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2025_11_29_scene_nomask_new"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2025_11_29_scene_nomask_new"
    
    # 文件名建议修改以体现实验性质
    val_filename = "test_tf_scene_nomask_d2_new1.txt"
    train_filename = "train_tf_scene_nomask_d2_new1.txt"
    fixed_prefix = "Data/ShapeNetP2M/"

    # 确保输出目录存在
    os.makedirs(output_meta_dir, exist_ok=True)
    train_dirs = []
    val_dirs = []

    # --- 2. 动态分组配置 (为了达到 2:1 比例) ---
    
    # 获取物体列表
    try:
        all_objects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        all_objects.sort() 
    except FileNotFoundError:
        print(f"错误：根目录 {root_dir} 未找到。")
        return

    # [物体划分] 5个训练 : 3个测试
    # 这样能保证模型学习到足够的物体特征
    OBJ_SPLIT_INDEX = 5 
    
    # 集合定义
    O_train_names = set(all_objects[:OBJ_SPLIT_INDEX]) # 物体 1 (O1-O5)
    O_test_names = set(all_objects[OBJ_SPLIT_INDEX:])  # 物体 2,3,4 (O6-O8)

    # [场景划分] 9个训练(A) : 8个测试(B)
    # 17个场景: 0-8 为 A类, 9-16 为 B类
    S_train_ids = list(range(0, 9))   # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    S_test_ids = list(range(9, 17))   # [9, 10, 11, 12, 13, 14, 15, 16]

    print("=== 双重未见 (Dual Unseen) 实验配置 ===")
    print(f"训练场景 (A类): {len(S_train_ids)} 个 -> {S_train_ids}")
    print(f"测试场景 (B类): {len(S_test_ids)} 个 -> {S_test_ids}")
    print(f"训练物体 (组1): {len(O_train_names)} 个 -> {sorted(list(O_train_names))}")
    print(f"测试物体 (组2): {len(O_test_names)} 个 -> {sorted(list(O_test_names))}")
    print("---------------------------------------")
    print(f"预计组合数 Train: {len(S_train_ids)*len(O_train_names)}")
    print(f"预计组合数 Test : {len(S_test_ids)*len(O_test_names)}")
    print("=======================================")

    # --- 3. 遍历与划分逻辑 ---
    for secondary_dir in os.listdir(root_dir):
        secondary_path = os.path.join(root_dir, secondary_dir)
        if not os.path.isdir(secondary_path):
            continue
        
        object_name = secondary_dir 

        # 遍历三级目录
        for tertiary_dir in os.listdir(secondary_path):
            tertiary_path = os.path.join(secondary_path, tertiary_dir)
            if not os.path.isdir(tertiary_path):
                continue
            
            try:
                parts = tertiary_dir.split('_')
                if len(parts) >= 2:
                    scene_id = int(parts[1]) # 解析场景ID
                    
                    # --- 核心逻辑: 严格隔离 ---
                    
                    # 情况 1: 训练集 (A场景 + 物体1)
                    if (scene_id in S_train_ids) and (object_name in O_train_names):
                        train_dirs.append((secondary_dir, tertiary_dir))
                        
                    # 情况 2: 测试集 (B场景 + 物体2,3,4)
                    # 注意: 这里既换了场景，也换了物体
                    elif (scene_id in S_test_ids) and (object_name in O_test_names):
                        val_dirs.append((secondary_dir, tertiary_dir))
                    
                    # 其他情况 (交叉组合) 全部丢弃，例如:
                    # - A场景 + 物体2 (场景见过，物体没见过 -> 半未见) -> 丢弃
                    # - B场景 + 物体1 (场景没见过，物体见过 -> 半未见) -> 丢弃
                    else:
                        pass 

            except ValueError:
                print(f"警告：无法解析场景ID: {parts[1]}")
            except Exception as e:
                print(f"错误：处理目录 {tertiary_path} 时出错: {e}")

    # 打乱顺序
    random.shuffle(train_dirs)
    random.shuffle(val_dirs)
    
    # 写入文件函数
    def write_paths(directories, output_file):
        with open(os.path.join(output_meta_dir, output_file), 'w', encoding='utf-8') as f:
            for secondary_dir, tertiary_dir in directories:
                mat_files_path = os.path.join(root_dir, secondary_dir, tertiary_dir)
                for i in range(4):
                    mat_file = os.path.join(mat_files_path, f"{i}.mat")
                    if os.path.exists(mat_file):
                        target_path = f"{fixed_prefix}{secondary_dir}/{tertiary_dir}/{i}.dat"
                        f.write(target_path + '\n')

    # 执行写入
    write_paths(val_dirs, val_filename)
    write_paths(train_dirs, train_filename)

    t_count = len(train_dirs)
    v_count = len(val_dirs)

    print(f"\n处理完成。")
    print(f"Train 条目数: {t_count}")
    print(f"Test  条目数: {v_count}")
    if v_count > 0:
        print(f"实际比例: {t_count / v_count:.2f} : 1")
    print(f"文件保存至: {output_meta_dir}")

if __name__ == "__main__":
    main()