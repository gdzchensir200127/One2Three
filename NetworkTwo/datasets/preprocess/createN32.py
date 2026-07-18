import os
import shutil
import re

def copy_and_rename_files(source_root, target_root, mask_suffix):
    """
    复制文件并按照规则重命名目录
    
    参数:
    source_root: 源目录根路径
    target_root: 目标目录根路径
    mask_suffix: 目录名后缀标识（_mask_ 或 _nomask_）
    """
    # 递归遍历源目录下的所有文件和目录
    for root, dirs, files in os.walk(source_root):
        # 跳过空目录
        if not files:
            continue
        
        # 计算当前路径相对于源根目录的相对路径
        relative_path = os.path.relpath(root, source_root)
        
        # 处理目录名：在 aug 前插入标识（例如 5_16_aug0 → 5_16_mask_aug0）
        # 使用正则表达式匹配并替换
        pattern = r'(\d+_\d+)_aug(\d+)'
        replacement = rf'\1{mask_suffix}aug\2'
        new_relative_path = re.sub(pattern, replacement, relative_path)
        
        # 构建目标目录路径
        target_dir = os.path.join(target_root, new_relative_path)
        
        # 创建目标目录（如果不存在），包括所有父目录
        os.makedirs(target_dir, exist_ok=True)
        
        # 复制所有文件到目标目录
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, file)
            
            # 复制文件（保留文件元数据）
            shutil.copy2(source_file, target_file)
            print(f"已复制: {source_file} → {target_file}")

if __name__ == "__main__":
    # 定义需要处理的物品列表
    item_list = ["手机", "键盘", "笔记本电脑", "水杯", "易拉罐", "显示器", "刀具", "扳手"]
    
    # 定义源目录和目标目录基础路径
    base_source = "/home/zhang_muxin/Signal2PC/datasets/data/N2_aug_base/"
    target_root_base = "/home/zhang_muxin/Signal2PC/datasets/data/N2_aug_base_base/"
    
    # 遍历每个物品进行处理
    for item in item_list:
        print(f"\n{'='*20} 开始处理【{item}】 {'='*20}")
        
        # 构建当前物品的目标根目录
        target_root = os.path.join(target_root_base, item)
        
        # 处理无遮挡文件（_nomask_）
        source_nomask = os.path.join(base_source, item)
        if os.path.exists(source_nomask):
            print(f"\n处理{item}无遮挡文件（源目录：{source_nomask}）...")
            copy_and_rename_files(source_nomask, target_root, "_nomask_")
        else:
            print(f"\n警告：{item}无遮挡源目录不存在 → {source_nomask}")
        
        # 处理有遮挡文件（_mask_）
        source_mask = os.path.join(base_source, f"{item}-遮挡")
        if os.path.exists(source_mask):
            print(f"\n处理{item}有遮挡文件（源目录：{source_mask}）...")
            copy_and_rename_files(source_mask, target_root, "_mask_")
        else:
            print(f"\n警告：{item}有遮挡源目录不存在 → {source_mask}")
    
    print(f"\n{'='*50}")
    print("所有物品处理完成！")