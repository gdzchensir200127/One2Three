import os
import shutil

def copy_folder_contents(source_dir, target_dir):
    """
    将源目录下的所有文件和子目录复制到目标目录
    保持原有的目录结构，若目标目录不存在则自动创建
    同名文件不会被覆盖，会输出提示信息并跳过
    
    Args:
        source_dir (str): 源目录路径
        target_dir (str): 目标目录路径
    """
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    
    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)
    print(f"目标目录准备完成: {target_dir}")
    
    # 遍历源目录下的所有文件和子目录（递归遍历所有层级）
    for root, dirs, files in os.walk(source_dir):
        # 计算当前目录相对于源目录的路径
        relative_path = os.path.relpath(root, source_dir)
        # 构建目标目录中的对应路径
        target_root = os.path.join(target_dir, relative_path)
        
        # 创建目标目录（如果不存在）
        os.makedirs(target_root, exist_ok=True)
        
        # 复制文件
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_root, file)
            
            try:
                # 检查目标文件是否已存在
                if os.path.exists(target_file):
                    print(f"⚠️  跳过已存在文件（不覆盖）: {target_file}")
                    continue
                
                # 复制文件（保留元数据）
                shutil.copy2(source_file, target_file)
                print(f"✅ 已复制文件: {source_file} -> {target_file}")
            
            except Exception as e:
                print(f"❌ 复制文件失败 {source_file}: {str(e)}")
        
        # 复制目录（其实上面的os.makedirs已经创建了目录结构，这里主要是输出提示）
        for dir_name in dirs:
            source_subdir = os.path.join(root, dir_name)
            target_subdir = os.path.join(target_root, dir_name)
            
            if not os.path.exists(target_subdir):
                os.makedirs(target_subdir, exist_ok=True)
                print(f"📂 已创建目录: {target_subdir}")
            else:
                print(f"📂 目录已存在（跳过创建）: {target_subdir}")
    
    print("\n📋 复制完成！已跳过所有同名文件，未覆盖任何原有文件。")

if __name__ == "__main__":
    # 定义源路径和目标路径
    source_directory = "/home/zhang_muxin/Signal2PC/datasets/data/N2_unseen_scene_base/test_aug/"
    target_directory = "/home/zhang_muxin/Signal2PC/datasets/data/N2_unseen_scene_base/train_aug/"
    
    try:
        copy_folder_contents(source_directory, target_directory)
    except Exception as main_e:
        print(f"❌ 执行失败: {str(main_e)}")