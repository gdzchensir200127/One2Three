import os
import shutil

def process_models_dirs(root_dir):
    """
    遍历根目录下所有models四级目录（根目录为0级，models为3级），执行删除和复制重命名操作
    修改后逻辑：
    1. 删除2.mat、3.mat（如果存在）
    2. 复制0.mat并重命名为2.mat
    3. 复制1.mat并重命名为3.mat（需先检查1.mat是否存在）
    :param root_dir: 一级根目录路径
    """
    # 确保根目录是绝对路径，避免相对路径问题
    root_dir = os.path.abspath(root_dir)
    print(f"根目录（绝对路径）: {root_dir}")

    # 遍历根目录下所有层级的目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 核心修正：计算相对于根目录的层级
        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path == '.':
            current_level = 0  # 根目录本身
        else:
            current_level = len(rel_path.split(os.sep))  # 拆分相对路径得层级
        
        # 判断条件：当前目录是models + 相对于根目录的层级是3（根=0级，二级=1，三级=2，models=3）
        is_models_dir = (os.path.basename(dirpath) == 'models') and (current_level == 3)
        
        if is_models_dir:
            print(f"\n===== 开始处理models目录: {dirpath} =====")
            
            # 构造关键文件路径
            mat0_path = os.path.join(dirpath, "0.mat")
            mat1_path = os.path.join(dirpath, "1.mat")
            
            # 检查0.mat是否存在（基础文件）
            if not os.path.exists(mat0_path):
                print(f"⚠️  警告：{dirpath} 中未找到0.mat，跳过该目录")
                continue

            # 1. 删除2.mat、3.mat（如果存在）
            deleted_count = 0
            for mat_num in [2, 3]:  # 仅删除2、3.mat
                mat_path = os.path.join(dirpath, f"{mat_num}.mat")
                if os.path.exists(mat_path):
                    try:
                        os.remove(mat_path)
                        print(f"✅ 已删除: {mat_path}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"❌ 删除 {mat_path} 失败: {e}")
            if deleted_count == 0:
                print("ℹ️  未找到2/3.mat，无需删除")
            
            # 2. 复制0.mat并重命名为2.mat
            copied_count = 0
            target_2mat = os.path.join(dirpath, "2.mat")
            try:
                shutil.copy2(mat0_path, target_2mat)  # 保留元数据
                print(f"✅ 已复制: {mat0_path} -> {target_2mat}")
                copied_count += 1
            except Exception as e:
                print(f"❌ 复制 {mat0_path} 到 {target_2mat} 失败: {e}")
            
            # 3. 复制1.mat并重命名为3.mat（先检查1.mat是否存在）
            if not os.path.exists(mat1_path):
                print(f"⚠️  警告：{dirpath} 中未找到1.mat，跳过3.mat的复制")
            else:
                target_3mat = os.path.join(dirpath, "3.mat")
                try:
                    shutil.copy2(mat1_path, target_3mat)
                    print(f"✅ 已复制: {mat1_path} -> {target_3mat}")
                    copied_count += 1
                except Exception as e:
                    print(f"❌ 复制 {mat1_path} 到 {target_3mat} 失败: {e}")
            
            # 输出处理完成状态
            if copied_count == 2:
                print(f"✅ {dirpath} 处理完成（2.mat和3.mat均复制成功）")
            elif copied_count == 1:
                print(f"⚠️ {dirpath} 部分完成（仅2.mat复制成功，3.mat未复制）")
            else:
                print(f"❌ {dirpath} 处理失败（2.mat和3.mat均未复制成功）")

if __name__ == "__main__":
    # 定义一级根目录（务必确认路径正确）
    ROOT_DIRECTORY = "/home/zhang_muxin/Signal2PC/datasets/data/N2_fsview"
    
    # 检查根目录是否存在
    if not os.path.exists(ROOT_DIRECTORY):
        print(f"❌ 错误：根目录 {ROOT_DIRECTORY} 不存在")
    else:
        print("开始处理所有models目录...")
        process_models_dirs(ROOT_DIRECTORY)
        print("\n===== 全局处理完成 =====")