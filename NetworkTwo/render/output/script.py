import os
import shutil
root_dir = "/home/zhang_muxin/Signal2mesh/render/output"
# 定义父目录列表
PARENT_DIRS = ["笔记本电脑", "水杯", "显示器", "扳手", "手机", "刀具", "键盘"]  # 替换为实际的父目录

def create_aug_dirs(parent_dirs):
    # 遍历每个父目录
    for parent_dir in parent_dirs:
        parent_dir = os.path.join(root_dir, parent_dir+"_depth_fixed4")
        # 检查父目录是否存在
        if not os.path.isdir(parent_dir):
            print(f"父目录 {parent_dir} 不存在，跳过")
            continue

        # 遍历父目录下的每个子目录
        for sub_dir in os.listdir(parent_dir):
            sub_dir_path = os.path.join(parent_dir, sub_dir)
            # 检查是否是目录
            if not os.path.isdir(sub_dir_path):
                file_path = sub_dir_path
                if file_path.endswith('.mat'):
                    file_name = file_path[:-4]
                    if os.path.basename(file_name).find('_') == -1:
                        for i in range(100):
                            aug_file_path = f"{file_name}_aug{i}.mat"
                            print(aug_file_path)
                            shutil.copyfile(file_path, aug_file_path)
            else:
                split_list = sub_dir.split('_')
                if len(split_list) > 2:
                    continue
                # 创建 50 个副本
                for i in range(100):
                    # 定义副本名称
                    aug_dir_name = f"{sub_dir}_aug{i}"
                    aug_dir_path = os.path.join(parent_dir, aug_dir_name)
                    if os.path.exists(aug_dir_path):
                        continue
                    # 复制子目录
                    shutil.copytree(sub_dir_path, aug_dir_path)

                    # 打印日志
                    print(f"已创建副本: {aug_dir_path}")

    print("所有副本创建完成")

# 执行函数
create_aug_dirs(PARENT_DIRS)