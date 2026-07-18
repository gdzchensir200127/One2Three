import os
import random

def main():
    # ================= 配置参数 =================
    # 请根据实际情况修改 root_dir
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/vas100/Scene_nomask_vas"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/o_s"
    
    val_filename = "test_tf_all_S2M_4views_pro_scenario_21.txt"
    train_filename = "train_tf_all_S2M_4views_pro_scenario_21.txt"
    fixed_prefix = "Data/ShapeNetP2M/"

    # 定义场景 ID
    # S1=14, S2=15, S3=16
    SCENE_S1_S2 = ['14', '15']
    SCENE_S3 = '16' # 单个字符串，或者用列表 ['16'] 也可以
    
    # 所有合法的场景用于初步筛选
    ALL_VALID_SCENES = ['14', '15', '16'] 
    
    # ===========================================

    # 确保输出目录存在
    os.makedirs(output_meta_dir, exist_ok=True)

    train_dirs = []
    val_dirs = []

    try:
        # 获取所有物体目录 (O1, O2... O8)
        all_objects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        all_objects.sort() # 排序保证 O1-O8 顺序固定
    except FileNotFoundError:
        print(f"错误：根目录 {root_dir} 未找到。")
        return

    if len(all_objects) < 8:
        print(f"警告：检测到 {len(all_objects)} 个物体，少于预期的 8 个。")

    # 定义物体分组
    # O1 ~ O4 (前4个) -> 训练候选
    O_train_objs = set(all_objects[:4]) 
    # O5 ~ O8 (后4个) -> 测试候选
    O_test_objs = set(all_objects[4:]) 

    print("=== 划分配置 ===")
    print(f"S1/S2 Scense: {SCENE_S1_S2}")
    print(f"S3 Scene:     {SCENE_S3}")
    print(f"Train Set Logic: (S1, S2)  * (O1-O4)")
    print(f"Test Set Logic:  (S3 Only) * (O5-O8)")
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
                # [已确认] 不过滤 mask/nomask，直接处理所有文件夹

                # 1. 解析场景 ID
                parts = tertiary_dir.split('_')
                if len(parts) < 2:
                    continue
                
                scene_id_str = parts[1] # 获取第二个部分作为场景ID

                # 2. 全局场景初步过滤：必须是 14, 15, 16 之一
                if scene_id_str not in ALL_VALID_SCENES:
                    continue

                # ================= 核心划分逻辑 =================
                
                # 逻辑 1: 训练集 Training Set
                # 规则: (s1,s2) * (o1~o4)
                if secondary_dir in O_train_objs:
                    # 必须属于 S1 或 S2 (14, 15)
                    if scene_id_str in SCENE_S1_S2:
                        train_dirs.append((secondary_dir, tertiary_dir))
                    # 注意：如果 scene_id_str 是 S3 (16)，这里会被忽略 (不加入训练集)

                # 逻辑 2: 测试集 Testing Set
                # 规则: (s3) * (o5~o8)
                elif secondary_dir in O_test_objs:
                    # 必须属于 S3 (16)
                    if scene_id_str == SCENE_S3:
                        val_dirs.append((secondary_dir, tertiary_dir))
                    # 注意：如果 scene_id_str 是 S1/S2，这里会被忽略

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
                        # 写入 Dat 路径 (包含 /models/)
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