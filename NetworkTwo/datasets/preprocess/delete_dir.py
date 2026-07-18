import os
import shutil
from pathlib import Path

def main():
    # 根路径（请再次确认路径正确性！）
    root_path = Path("/home/zhang_muxin/Signal2PC/datasets/data/N2_trained_base/test")
    
    # 安全检查：确认根路径存在
    if not root_path.exists() or not root_path.is_dir():
        print(f"错误：根路径不存在或不是目录 -> {root_path}")
        return
    
    # 存储要删除的目录列表（用于预览）
    to_delete = []
    
    # 遍历一级目录
    for level1 in root_path.iterdir():
        if not level1.is_dir():
            print(f"跳过非目录文件（一级）：{level1}")
            continue
        
        # 遍历二级目录
        for level2 in level1.iterdir():
            if not level2.is_dir():
                print(f"跳过非目录文件（二级）：{level2}")
                continue
            
            # 遍历三级目录
            for level3 in level2.iterdir():
                if not level3.is_dir():
                    print(f"跳过非目录文件（三级）：{level3}")
                    continue
                
                # 检查三级目录名称是否为"16"
                if level3.name != "16":
                    to_delete.append(level3)
    
    # 预览结果
    print("\n" + "="*50)
    print(f"扫描完成！找到 {len(to_delete)} 个需要删除的三级目录：")
    print("="*50)
    for idx, dir_path in enumerate(to_delete, 1):
        print(f"{idx:3d}. {dir_path}")
    
    # 确认是否执行删除
    if not to_delete:
        print("\n没有需要删除的目录，程序退出。")
        return
    
    confirm = input("\n是否确认删除以上所有目录及下属文件？(y/N)：").strip().lower()
    if confirm != "y":
        print("用户取消删除操作，程序退出。")
        return
    
    # 执行删除
    print("\n开始执行删除操作...")
    deleted_count = 0
    failed_count = 0
    failed_dirs = []
    
    for dir_path in to_delete:
        try:
            # 强制删除目录树（包括所有文件和子目录）
            shutil.rmtree(dir_path)
            print(f"✅ 已删除：{dir_path}")
            deleted_count += 1
        except Exception as e:
            print(f"❌ 删除失败：{dir_path} -> 错误：{str(e)}")
            failed_count += 1
            failed_dirs.append((dir_path, str(e)))
    
    # 输出执行结果汇总
    print("\n" + "="*50)
    print("删除操作完成！")
    print("="*50)
    print(f"总扫描到需要删除的目录数：{len(to_delete)}")
    print(f"成功删除目录数：{deleted_count}")
    print(f"删除失败目录数：{failed_count}")
    
    if failed_dirs:
        print("\n删除失败的目录详情：")
        for dir_path, error in failed_dirs:
            print(f"  - {dir_path}：{error}")

if __name__ == "__main__":
    main()