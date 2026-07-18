import os
import argparse

def generate_category_lists(root_dir):
    # 遍历 root_dir 下的所有子目录（每个子目录代表一个「种类」）
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        
        # 确保是目录且非隐藏文件（可选）
        if os.path.isdir(category_path) and not category.startswith('.'):
            list_filename = f"{category}.list"
            
            # 获取该「种类」目录下的所有物体（子目录）
            objects = []
            for obj in os.listdir(category_path):
                obj_path = os.path.join(category_path, obj)
                if os.path.isdir(obj_path):  # 只处理子目录
                    objects.append(obj)
            
            # 写入 .list 文件
            with open(list_filename, 'w') as f:
                f.write('\n'.join(objects))
            print(f"生成文件: {list_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成种类对应的 .list 文件')
    parser.add_argument('root_dir', help='根目录路径（例如: root_dir/）')
    args = parser.parse_args()
    
    if not os.path.exists(args.root_dir):
        print(f"错误：目录 {args.root_dir} 不存在")
    else:
        generate_category_lists(args.root_dir)