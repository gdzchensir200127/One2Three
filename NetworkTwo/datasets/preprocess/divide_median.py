import os

def filter_dataset(input_path, output_path, target_category, target_style):
    # 1. 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到输入文件，请检查路径是否正确：\n{input_path}")
        return

    # 2. 检查输出路径的文件夹是否存在，如果不存在则自动创建
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"📂 已自动创建输出目录：{output_dir}")
        except OSError as e:
            print(f"❌ 创建目录失败：{e}")
            return

    target_style = str(target_style)
    count = 0
    print(f"正在读取文件：{input_path}")
    print(f"筛选条件 -> 类别: [{target_category}], 样式ID: [{target_style}]")

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                clean_line = line.strip()
                if not clean_line:
                    continue
                
                # 路径示例: Data/ShapeNetP2M/键盘/1_3_nomask_aug8/models/0.dat
                parts = clean_line.split('/')
                
                # 安全检查：防止空行或格式错误的行导致崩溃
                if len(parts) < 4:
                    continue

                category = parts[2]       # 获取 "键盘"
                folder_name = parts[3]    # 获取 "1_3_nomask_aug8"
                
                # 提取样式ID (1_3... -> 取 1)
                try:
                    style_id = folder_name.split('_')[0]
                except IndexError:
                    continue 
                
                # 匹配逻辑
                if category == target_category and style_id == target_style:
                    f_out.write(clean_line + '\n')
                    count += 1

        print(f"✅ 筛选完成！")
        print(f"📊 共找到 {count} 条数据")
        print(f"💾 结果已保存至：{output_path}")

    except Exception as e:
        print(f"❌ 发生未知错误：{e}")

# ==================== 配置区域 (请在此处修改路径) ====================

# 1. 输入文件的完整路径 (如果是 Windows 路径请注意斜杠，推荐使用 Linux 风格的反斜杠 /)
# 例如: '/home/user/data/dataset_lists/full_train_list.txt'
input_txt_path = '/home/zhang_muxin/Signal2PC/datasets/data/N33/meta/val_tf_all_S2M_4views_pro_N33dataset.txt'

# 2. 输出文件的完整路径
# 例如: '/home/user/projects/experiment_1/keyboard_style_1.txt'
output_txt_path = '/home/zhang_muxin/Signal2PC/datasets/data/N33/meta/val_tf_all_S2M_4views_pro_易拉罐6.txt'

# 3. 筛选目标
target_category = '易拉罐'   # 目标类别
target_style =6          # 目标样式 (1-6)

# ===================================================================

if __name__ == '__main__':
    filter_dataset(input_txt_path, output_txt_path, target_category, target_style)