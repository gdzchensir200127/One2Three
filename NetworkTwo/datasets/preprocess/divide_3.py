import os
import random


def main():
    # ================= 配置参数 =================
    root_dir = "/home/zhang_muxin/Signal2PC/datasets/data/scene_nomask_select"
    output_meta_dir = "/home/zhang_muxin/Signal2PC/datasets/data/scene_nomask_baseline/meta"

    val_filename = "val_tf_scene_nomask_baseline.txt"
    train_filename = "train_tf_scene_nomask_baseline.txt"
    test_filename = "test_tf_scene_nomask_baseline.txt"

    fixed_prefix = "Data/ShapeNetP2M/"
    # ===========================================

    # 确保输出目录存在
    os.makedirs(output_meta_dir, exist_ok=True)

    # ---------------------------------------------
    # 第一步：收集所有二级目录（作为划分的最小单位）
    # ---------------------------------------------
    # 假设二级目录代表“物体”或“类别”（如扳手、显示器），
    # 我们要保证同一个二级目录下的所有数据只存在于一个集合中。

    all_secondary_dirs = []
    if os.path.exists(root_dir):
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                all_secondary_dirs.append(item)
    else:
        print(f"错误：根目录不存在 {root_dir}")
        return

    # 检查是否找到数据
    total_objects = len(all_secondary_dirs)
    if total_objects == 0:
        print("未找到任何二级目录，请检查路径。")
        return

    # ---------------------------------------------
    # 第二步：对二级目录进行 4:1:1 划分
    # ---------------------------------------------
    # 计算各部分的数量
    unit_count = total_objects // 6

    test_count = unit_count  # 1份
    val_count = unit_count  # 1份
    # 剩余的给训练集（约4份），这样可以处理除不尽的情况
    train_count = total_objects - val_count - test_count

    # 随机打乱“物体/类别”列表
    random.shuffle(all_secondary_dirs)

    # 切分列表
    test_objs = all_secondary_dirs[:test_count]
    remaining_objs = all_secondary_dirs[test_count:]

    val_objs = remaining_objs[:val_count]
    train_objs = remaining_objs[val_count:]

    # 将列表转为集合(Set)，方便后续快速判断
    test_obj_set = set(test_objs)
    val_obj_set = set(val_objs)
    train_obj_set = set(train_objs)

    print(f"--- 物体/类别划分统计 ---")
    print(f"总物体数: {total_objects}")
    print(f"Train物体: {len(train_objs)}")
    print(f"Val物体:   {len(val_objs)}")
    print(f"Test物体:  {len(test_objs)}")
    print(f"-------------------------")

    # ---------------------------------------------
    # 第三步：遍历所有数据，根据所属二级目录分配到对应集合
    # ---------------------------------------------
    train_paths = []
    val_paths = []
    test_paths = []

    # 再次遍历目录结构来收集具体的文件路径
    for sec_dir in all_secondary_dirs:
        sec_path = os.path.join(root_dir, sec_dir)

        # 确定当前二级目录属于哪个集合
        if sec_dir in train_obj_set:
            target_list = train_paths
        elif sec_dir in val_obj_set:
            target_list = val_paths
        elif sec_dir in test_obj_set:
            target_list = test_paths
        else:
            continue  # 理论上不会执行到这里

        # 遍历该物体下的三级目录（实例/视角）
        if os.path.isdir(sec_path):
            for tert_dir in os.listdir(sec_path):
                tert_path = os.path.join(sec_path, tert_dir)
                if not os.path.isdir(tert_path):
                    continue

                # 保存 (二级目录, 三级目录) 元组，稍后用于生成路径
                target_list.append((sec_dir, tert_dir))

    # ---------------------------------------------
    # 第四步：写入文件
    # ---------------------------------------------
    def write_to_file(data_list, filename):
        full_path = os.path.join(output_meta_dir, filename)
        count = 0
        with open(full_path, 'w', encoding='utf-8') as f:
            for sec_dir, tert_dir in data_list:
                # 检查4个mat文件是否存在
                mat_base_path = os.path.join(root_dir, sec_dir, tert_dir)
                for i in range(4):
                    mat_file = os.path.join(mat_base_path, f"{i}.mat")
                    if os.path.exists(mat_file):
                        # 按要求格式写入
                        line = f"{fixed_prefix}{sec_dir}/{tert_dir}/{i}.dat"
                        f.write(line + '\n')
                        count += 1
        return count

    # 执行写入
    n_train = write_to_file(train_paths, train_filename)
    n_val = write_to_file(val_paths, val_filename)
    n_test = write_to_file(test_paths, test_filename)

    print(f"\n--- 最终数据文件统计 (行数) ---")
    print(f"Train 数据条目数: {n_train}")
    print(f"Val   数据条目数: {n_val}")
    print(f"Test  数据条目数: {n_test}")
    print(f"元数据文件已保存至: {output_meta_dir}")


if __name__ == "__main__":
    main()