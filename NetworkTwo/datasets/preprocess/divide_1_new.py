import os
import random
import shutil

def main():
    # --- 基础路径配置 ---
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2025_11_29_scene_nomask_new"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/aug_2025_11_29_scene_nomask_new"
    
    # 输出文件名
    val_filename = "test_tf_scene_nomask_d1_new1.txt"
    train_filename = "train_tf_scene_nomask_d1_new1.txt"
    fixed_prefix = "Data/ShapeNetP2M/"

    # ================= 配置区域 (已调整为 2:1 比例) =================

    # 1. 物体分组 (6个物体进组1，2个物体进组2)
    # 组1: O1-O6 (训练主力)
    # 组2: O7-O8 (测试主力)
    SPLIT_INDEX = 6 

    # 2. 场景分组 (17个场景)
    
    # 【A类场景】：Train 见 组1 (6个物体), Test 见 组2 (2个物体)
    # 这里放了 15 个场景，以最大化训练集数量
    SCENES_TYPE_A = list(range(0, 17)) # [0, 1, ... 14]

    # 【B类场景】：Train 见 组2 (2个物体), Test 见 组1 (6个物体)
    # 这里只放 1 个场景，用来证明模型能处理反转组合，但不会过多稀释训练集
    SCENES_TYPE_B = []

    # 【C类场景】：完全未见场景
    # 放在测试集
    SCENES_TYPE_C = []

    # ===========================================

    os.makedirs(output_meta_dir, exist_ok=True)
    train_dirs = []
    val_dirs = []

    # 获取并排序物体列表
    try:
        all_objects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        all_objects.sort() 
    except FileNotFoundError:
        print(f"错误：根目录 {root_dir} 未找到。")
        return

    # 划分物体组
    obj_group_1 = set(all_objects[:SPLIT_INDEX]) # 大组 (6个)
    obj_group_2 = set(all_objects[SPLIT_INDEX:]) # 小组 (2个)

    print("=== 2:1 比例实验配置确认 ===")
    print(f"物体组1 (Train for A): {len(obj_group_1)} 个 -> {sorted(list(obj_group_1))}")
    print(f"物体组2 (Train for B): {len(obj_group_2)} 个 -> {sorted(list(obj_group_2))}")
    print(f"A类场景 (数量 {len(SCENES_TYPE_A)}): {SCENES_TYPE_A}")
    print(f"B类场景 (数量 {len(SCENES_TYPE_B)}): {SCENES_TYPE_B}")
    print(f"C类场景 (数量 {len(SCENES_TYPE_C)}): {SCENES_TYPE_C}")
    print("==========================")

    # 遍历处理
    for secondary_dir in os.listdir(root_dir):
        secondary_path = os.path.join(root_dir, secondary_dir)
        if not os.path.isdir(secondary_path):
            continue

        object_name = secondary_dir 

        for tertiary_dir in os.listdir(secondary_path):
            tertiary_path = os.path.join(secondary_path, tertiary_dir)
            if not os.path.isdir(tertiary_path):
                continue
            
            try:
                parts = tertiary_dir.split('_')
                if len(parts) >= 2:
                    scene_id = int(parts[1])

                    # --- 核心划分逻辑 ---

                    # 逻辑 1: A类场景 (Train: Group 1 / Test: Group 2)
                    if scene_id in SCENES_TYPE_A:
                        if object_name in obj_group_1:
                            train_dirs.append((secondary_dir, tertiary_dir))
                        elif object_name in obj_group_2:
                            val_dirs.append((secondary_dir, tertiary_dir))

                    # 逻辑 2: B类场景 (Train: Group 2 / Test: Group 1)
                    elif scene_id in SCENES_TYPE_B:
                        if object_name in obj_group_2:
                            train_dirs.append((secondary_dir, tertiary_dir))
                        elif object_name in obj_group_1:
                            val_dirs.append((secondary_dir, tertiary_dir))

                    # 逻辑 3: C类场景 (Test: All)
                    elif scene_id in SCENES_TYPE_C:
                        val_dirs.append((secondary_dir, tertiary_dir))
                    
                    # -------------------

            except ValueError:
                pass 
            except Exception as e:
                print(f"错误: {e}")

    # 打乱
    random.shuffle(train_dirs)
    random.shuffle(val_dirs)
    
    # 写入函数
    def write_paths(directories, output_file):
        with open(os.path.join(output_meta_dir, output_file), 'w', encoding='utf-8') as f:
            for secondary_dir, tertiary_dir in directories:
                mat_files_path = os.path.join(root_dir, secondary_dir, tertiary_dir)
                for i in range(4): 
                    mat_file = os.path.join(mat_files_path, f"{i}.mat")
                    if os.path.exists(mat_file):
                        target_path = f"{fixed_prefix}{secondary_dir}/{tertiary_dir}/{i}.dat"
                        f.write(target_path + '\n')

    write_paths(val_dirs, val_filename)
    write_paths(train_dirs, train_filename)

    t_count = len(train_dirs)
    v_count = len(val_dirs)
    
    print(f"\n生成完成。")
    print(f"Train 条目数: {t_count}")
    print(f"Test  条目数: {v_count}")
    if v_count > 0:
        print(f"最终比例: {t_count / v_count:.2f} : 1")

if __name__ == "__main__":
    main()