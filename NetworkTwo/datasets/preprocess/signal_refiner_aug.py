import os
import numpy as np
from scipy.io import loadmat, savemat
import argparse

def random_shift_matrix(data:np.ndarray, shift, target_frame):
    end_index = min(data.shape[0], shift+target_frame-1)
    shifted_data = data[end_index-target_frame:end_index]
    return shifted_data, shift

def sync_augmentation(params):
    """
    参数：
    params: 包含所有配置参数的字典
    """
    # 参数校验
    assert os.path.exists(params['root_dir1']), f"Directory {params['root_dir1']} not found"
    assert os.path.exists(params['root_dir2']), f"Directory {params['root_dir2']} not found"
    assert params['num_augments'] > 0, "num_augments must be positive"
    assert 1 <= params['max_shift'] <= 2800, "max_shift must be between 1-2800"

    # 创建种子以保证可重复性
    if params['seed'] is not None:
        np.random.seed(params['seed'])

    for category in os.listdir(params['root_dir1']):
        # 同步遍历两个数据集
        path_pair = (
            os.path.join(params['root_dir1'], category),
            os.path.join(params['root_dir2'], category)
        )

        # 跳过非目录项
        if not all(os.path.isdir(p) for p in path_pair):
            continue

        for obj_id in os.listdir(path_pair[0]):
            # 构建双路径
            obj_paths = [
                os.path.join(p, obj_id, 'models') 
                for p in path_pair
            ]

            # 验证路径一致性
            if not all(os.path.isdir(p) for p in obj_paths):
                print(obj_paths)
                print(f"Mismatch paths in {category} {obj_id}, skipping...")
                continue

            # 获取匹配的文件列表
            files1 = set(os.listdir(obj_paths[0]))
            files2 = set(os.listdir(obj_paths[1]))
            common_files = files1 & files2
            for mat_file in common_files:
                # 加载双数据集数据
                data_pair = []
                for p in obj_paths:
                    mat_path = os.path.join(p, mat_file)
                    try:
                        data_pair.append(loadmat(mat_path)['data'])
                    except Exception as e:
                        print(f"Error loading {mat_path}: {str(e)}")
                        continue
                # 生成增强数据
                for aug_idx in range(params['num_augments']):
                    # 生成相同的随机平移量
                    shifted_data = []
                    max_shift = params['max_shift']
                    shift = np.random.randint(0, max_shift)
                    for data in data_pair:
                        sd, shift = random_shift_matrix(data, shift, params['target_frame'])
                        shifted_data.append(sd)
                    # 保存到双数据集
                    for dataset_idx, (new_root, data) in enumerate(zip(
                        [params['new_root1'], params['new_root2']],
                        shifted_data
                    )):
                        new_obj_id = f"{obj_id}_arg{aug_idx}"
                        save_path = os.path.join(
                            new_root,
                            category,
                            new_obj_id,
                            'models'
                        )
                        os.makedirs(save_path, exist_ok=True)
                        
                        savemat(
                            os.path.join(save_path, mat_file),
                            {'data': data, 'aug_shift': shift}
                        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='双数据集同步数据增强')
    parser.add_argument('--root_dir1', type=str, required=True, help='原始数据集1路径')
    parser.add_argument('--root_dir2', type=str, required=True, help='原始数据集2路径')
    parser.add_argument('--new_root1', type=str, required=True, help='新数据集1保存路径')
    parser.add_argument('--new_root2', type=str, required=True, help='新数据集2保存路径')
    parser.add_argument('--num_augments', type=int, default=100, help='每个样本的增强次数')
    parser.add_argument('--target_frame', type=int, default=2800, help='目标信号长度')
    parser.add_argument('--max_shift', type=int, default=80, help='最大平移量（0-100）')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（可选）')
    
    args = parser.parse_args()
    
    params = {
        'root_dir1': args.root_dir1,
        'root_dir2': args.root_dir2,
        'new_root1': args.new_root1,
        'new_root2': args.new_root2,
        'num_augments': args.num_augments,
        'target_frame': args.target_frame,
        'max_shift': args.max_shift,
        'seed': args.seed
    }
    print(params)
    
    sync_augmentation(params)