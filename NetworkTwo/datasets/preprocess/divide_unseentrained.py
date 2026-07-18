import os
import random

def main():
    # ================= 配置参数 =================
    # 请根据实际情况修改 root_dir
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/vas100/Scene_nomask_vas"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/o_s/meta"
    
    val_filename = "test_tf_all_S2M_4views_pro_unseentrained2.txt"
    train_filename = "train_tf_all_S2M_4views_pro_unseentrained2.txt"
    fixed_prefix = "Data/ShapeNetP2M/"

    # 场景定义
    SCENE_S1_S2 = ['14', '15']
    SCENE_S3 = ['16']
    
    # ===========================================

    # 确保输出目录存在
    os.makedirs(output_meta_dir, exist_ok=True)

    train_dirs = []
    val_dirs = []

    try:
        # 获取所有物体目录
        all_objects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        all_objects.sort() # 排序保证 O1-O8 顺序固定
    except FileNotFoundError:
        print(f"错误：根目录 {root_dir} 未找到。")
        return

    if len(all_objects) < 8:
        print(f"警告：检测到 {len(all_objects)} 个物体，少于预期的 8 个，划分可能不准确。")

    # 定义物体分组
    # O1 ~ O4 (前4个)
    O_train_objs = set(all_objects[:4]) 
    # O5 ~ O8 (后4个)
    O_test_objs = set(all_objects[4:]) 

    print("=== 划分配置 ===")
    print(f"S1/S2 (场景 {SCENE_S1_S2})")
    print(f"S3    (场景 {SCENE_S3})")
    print(f"Train Objects (O1-O4): {sorted(list(O_train_objs))}")
    print(f"Test Objects  (O5-O8): {sorted(list(O_test_objs))}")
    print("-" * 30)

    # 遍历物体目录 (一级)
    for secondary_dir in os.listdir(root_dir):
        secondary_path = os.path.join(root_dir, secondary_dir)
        if not os.path.isdir(secondary_path):
            continue

        # 遍历具体数据目录 (二级)
        for tertiary_dir in os.listdir(secondary_path):
            tertiary_path = os.path.join(secondary_path, tertiary_dir)
            if not os.path.isdir(tertiary_path):
                continue
            
            try:
                # [已删除] 过滤 Mask 的逻辑
                # 原代码: if 'nomask' not in tertiary_dir: continue

                # 1. 解析场景 ID
                # 格式示例：6_16_nomask_aug1 -> 分割后 ['6', '16', 'nomask', 'aug1']
                parts = tertiary_dir.split('_')
                if len(parts) < 2:
                    continue
                
                scene_id_str = parts[1] # 获取第二个部分作为场景ID

                # 2. 场景过滤：只保留 14, 15, 16
                if scene_id_str not in SCENE_S1_S2 and scene_id_str not in SCENE_S3:
                    continue

                # ================= 核心划分逻辑 =================
                
                # 情况 A: 场景 S1, S2 (14, 15)
                if scene_id_str in SCENE_S1_S2:
                    # 规则: (s1,s2) * (o1~o4) -> Train
                    if secondary_dir in O_train_objs:
                        train_dirs.append((secondary_dir, tertiary_dir))
                    # (s1,s2) * (o5~o8) -> 未定义，忽略

                # 情况 B: 场景 S3 (16)
                elif scene_id_str in SCENE_S3:
                    # 规则: (s3) * (o5~o8) -> Test
                    if secondary_dir in O_test_objs:
                        val_dirs.append((secondary_dir, tertiary_dir))
                    
                    # 规则修正: (s3) * (o1~o4) -> Train (Unseen Object Setting)
                    elif secondary_dir in O_train_objs:
                        train_dirs.append((secondary_dir, tertiary_dir))

            except Exception as e:
                print(f"错误处理 {tertiary_dir}: {e}")

    # 打乱顺序
    random.shuffle(train_dirs)
    random.shuffle(val_dirs)

    # 写入文件函数
    def write_paths(directories, output_file):
        with open(os.path.join(output_meta_dir, output_file), 'w', encoding='utf-8') as f:
            for secondary_dir, tertiary_dir in directories:
                # 路径: root / object / folder / models / 0.mat
                mat_files_base = os.path.join(root_dir, secondary_dir, tertiary_dir, "models")

                for i in range(4):
                    mat_file = os.path.join(mat_files_base, f"{i}.mat")
                    # 检查文件是否存在
                    if os.path.exists(mat_file):
                        # 写入 Dat 路径 (包含 models)
                        target_path = f"{fixed_prefix}{secondary_dir}/{tertiary_dir}/models/{i}.dat"
                        f.write(target_path + '\n')

    write_paths(val_dirs, val_filename)
    write_paths(train_dirs, train_filename)

    print(f"划分完成！")
    print(f"Train 样本目录数: {len(train_dirs)}")
    print(f"Test  样本目录数: {len(val_dirs)}")
    print(f"文件保存至: {output_meta_dir}")

if __name__ == "__main__":
    main()