import os
import shutil

def process_models_dirs(root_dir):
    """
    遍历根目录下所有models四级目录（根目录为0级，models为3级），执行删除和复制重命名操作
    :param root_dir: 一级根目录路径
    操作逻辑：
    1. 删除目录下的1.mat、3.mat（如果存在）
    2. 复制0.mat并重命名为1.mat
    3. 复制2.mat并重命名为3.mat（若2.mat不存在则提示警告）
    """
    # 确保根目录是绝对路径，避免相对路径问题
    root_dir = os.path.abspath(root_dir)
    print(f"根目录（绝对路径）: {root_dir}")

    # 遍历根目录下所有层级的目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 计算相对于根目录的层级
        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path == '.':
            current_level = 0  # 根目录本身
        else:
            current_level = len(rel_path.split(os.sep))  # 拆分相对路径得层级
        
        # 判断条件：当前目录是models + 相对于根目录的层级是3（根=0级，二级=1，三级=2，models=3）
        is_models_dir = (os.path.basename(dirpath) == 'models') and (current_level == 3)
        
        if is_models_dir:
            print(f"\n===== 开始处理models目录: {dirpath} =====")
            
            # 构造0.mat的路径
            mat0_path = os.path.join(dirpath, "0.mat")
            # 检查0.mat是否存在（因为要复制到1.mat）
            if not os.path.exists(mat0_path):
                print(f"⚠️  警告：{dirpath} 中未找到0.mat，跳过该目录")
                continue
            
            # 1. 删除1.mat、3.mat（如果存在）
            deleted_count = 0
            for mat_num in [1, 3]:  # 仅删除1和3.mat
                mat_path = os.path.join(dirpath, f"{mat_num}.mat")
                if os.path.exists(mat_path):
                    try:
                        os.remove(mat_path)
                        print(f"✅ 已删除: {mat_path}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"❌ 删除 {mat_path} 失败: {e}")
            if deleted_count == 0:
                print("ℹ️  未找到1/3.mat，无需删除")
            elif deleted_count == 1:
                print(f"ℹ️  已删除{deleted_count}个文件（1/3.mat）")
            
            # 2. 执行复制重命名操作
            copied_count = 0
            
            # 2.1 复制0.mat -> 1.mat
            target_1mat = os.path.join(dirpath, "1.mat")
            try:
                shutil.copy2(mat0_path, target_1mat)  # 保留元数据
                print(f"✅ 已复制: {mat0_path} -> {target_1mat}")
                copied_count += 1
            except Exception as e:
                print(f"❌ 复制 {mat0_path} 到 {target_1mat} 失败: {e}")
            
            # 2.2 复制2.mat -> 3.mat（先检查2.mat是否存在）
            mat2_path = os.path.join(dirpath, "2.mat")
            target_3mat = os.path.join(dirpath, "3.mat")
            if not os.path.exists(mat2_path):
                print(f"⚠️  警告：{dirpath} 中未找到2.mat，无法完成2.mat -> 3.mat的复制")
            else:
                try:
                    shutil.copy2(mat2_path, target_3mat)
                    print(f"✅ 已复制: {mat2_path} -> {target_3mat}")
                    copied_count += 1
                except Exception as e:
                    print(f"❌ 复制 {mat2_path} 到 {target_3mat} 失败: {e}")
            
            # 输出复制操作结果
            if copied_count == 2:
                print(f"✅ {dirpath} 处理完成（所有复制操作成功）")
            elif copied_count == 1:
                print(f"⚠️  {dirpath} 部分完成（仅{copied_count}个复制操作成功）")
            else:
                print(f"❌ {dirpath} 复制操作全部失败")

if __name__ == "__main__":
    # 定义一级根目录（务必确认路径正确）
    ROOT_DIRECTORY = "/home/zhang_muxin/Signal2PC/datasets/data/N2_fbview"
    
    # 检查根目录是否存在
    if not os.path.exists(ROOT_DIRECTORY):
        print(f"❌ 错误：根目录 {ROOT_DIRECTORY} 不存在")
    else:
        print("开始处理所有models目录...")
        process_models_dirs(ROOT_DIRECTORY)
        print("\n===== 全局处理完成 =====")