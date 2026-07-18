import os
import shutil

def copy_aug0_directories(root_dir):
    """
    遍历指定根目录下的所有三级目录（以_aug0结尾），并复制生成aug1到aug24的副本
    
    Args:
        root_dir (str): 一级目录的绝对路径
    """
    # 验证根目录是否存在
    if not os.path.isdir(root_dir):
        print(f"错误：根目录 {root_dir} 不存在！")
        return

    # 遍历一级目录下的所有二级目录
    for second_level_name in os.listdir(root_dir):
        second_level_path = os.path.join(root_dir, second_level_name)
        
        # 跳过非目录文件（确保只处理二级目录）
        if not os.path.isdir(second_level_path):
            print(f"跳过非目录文件：{second_level_path}")
            continue

        # 遍历二级目录下的所有三级目录
        for third_level_name in os.listdir(second_level_path):
            third_level_path = os.path.join(second_level_path, third_level_name)
            
            # 筛选出以_aug0结尾的三级目录
            if os.path.isdir(third_level_path) and third_level_name.endswith("_aug0"):
                print(f"\n开始处理源目录：{third_level_path}")
                
                # 提取基础名称（去掉_aug0后缀）
                base_name = third_level_name[:-len("_aug0")]
                
                # 生成aug1到aug24的副本目录
                for aug_num in range(1, 25):
                    # 构建目标目录名称和路径
                    target_dir_name = f"{base_name}_aug{aug_num}"
                    target_dir_path = os.path.join(second_level_path, target_dir_name)
                    
                    # 检查目标目录是否已存在（避免重复复制）
                    if os.path.exists(target_dir_path):
                        print(f"跳过：目标目录 {target_dir_path} 已存在")
                        continue
                    
                    try:
                        # 复制目录及其所有内容（dirs_exist_ok=True兼容Python3.8+）
                        shutil.copytree(
                            src=third_level_path,
                            dst=target_dir_path,
                            dirs_exist_ok=True  # 若目标目录已存在则不报错（防止意外中断后重复执行）
                        )
                        print(f"成功复制：{target_dir_path}")
                    
                    except PermissionError:
                        print(f"权限错误：无法复制 {target_dir_path}（请检查文件权限）")
                    except Exception as e:
                        print(f"复制失败 {target_dir_path}：{str(e)}")

if __name__ == "__main__":
    # 定义一级目录路径
    ROOT_DIRECTORY = "/home/zhang_muxin/Signal2PC/datasets/data/N2_noaug"
    
    # 执行复制任务
    copy_aug0_directories(ROOT_DIRECTORY)
    
    print("\n===== 目录复制任务执行完成 =====")