def keep_first_n_lines(file_path, n=2612, encoding='utf-8'):
    """
    保留txt文件的前n行，删除其余行
    
    参数:
        file_path: str - 目标txt文件的路径（绝对路径或相对路径）
        n: int - 要保留的行数（默认2612行）
        encoding: str - 文件编码（默认utf-8，如遇乱码可尝试'gbk'）
    """
    try:
        # 第一步：读取文件前n行
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            lines = f.readlines()  # 读取所有行到列表
        
        # 检查文件总行数是否小于等于n
        total_lines = len(lines)
        if total_lines <= n:
            print(f"✅ 文件总行数({total_lines})不超过{n}行，无需修改")
            return
        
        # 第二步：只保留前n行并写回文件（覆盖原文件）
        with open(file_path, 'w', encoding=encoding) as f:
            f.writelines(lines[:n])  # 写入前n行
        
        print(f"✅ 操作成功！")
        print(f"📊 原文件总行数：{total_lines}")
        print(f"📊 保留行数：{n}")
        print(f"📊 删除行数：{total_lines - n}")
        
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{file_path}'，请检查文件路径是否正确")
    except PermissionError:
        print(f"❌ 错误：没有文件操作权限，请关闭正在占用该文件的程序")
    except Exception as e:
        print(f"❌ 操作失败：{str(e)}")


# ------------------- 使用说明 -------------------
if __name__ == "__main__":
    # 请修改下面的文件路径为你的txt文件路径
    # 示例1：相对路径（文件和脚本在同一文件夹）
    # target_file = "你的文件.txt"
    
    # 示例2：绝对路径（Windows系统）
    # target_file = r"C:\Users\你的用户名\Documents\目标文件.txt"
    
    # 示例3：绝对路径（Mac/Linux系统）
    # target_file = "/Users/你的用户名/Documents/目标文件.txt"
    
    target_file = "/home/zhang_muxin/Signal2PC/datasets/data/N33/meta/train_tf_all_S2M_4views_pro_N33dataset_70percent.txt"  # ← 关键：修改这里的文件路径
    
    # 执行操作（默认保留前2612行，如需修改行数可调整第二个参数）
    keep_first_n_lines(target_file, n=36560)