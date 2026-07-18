import os
import scipy.io as scio
import torch
import numpy as np
from datasets.base_dataset import BaseDataset

class Signal2PixelInferenceDataset(BaseDataset):
    def __init__(self, file_root, file_list_name=None):
        super().__init__()
        self.file_root = file_root
        
        # 1. 自动扫描文件夹下的所有 .mat 文件
        # 假设文件夹里正好放着那 4 个文件
        all_files = [f for f in os.listdir(self.file_root) if f.endswith('.mat')]
        all_files.sort() # 排序，确保顺序是 0, 1, 2, 3
        
        # 检查文件数量
        if len(all_files) != 4:
            print(f"Warning: Found {len(all_files)} files in {self.file_root}. Expected 4 views.")
            # 如果你不介意，也可以让它重复读取，但最好是确保有4个
        
        self.view_files = all_files
        
        # 为了让 DataLoader 运行，我们需要定义“有多少个样本”
        # 因为你只想推理这一组数据，所以样本数是 1
        self.file_names = ["inference_sample"] 

    def __getitem__(self, index):
        # index 在这里其实没用了，因为我们只有一组数据
        
        signals_list = []
        
        # 2. 循环读取 4 个文件，构建模型需要的 List
        for i in range(4):
            # 如果文件少于4个，就循环利用；如果正好4个，就一一对应
            file_name = self.view_files[i % len(self.view_files)]
            file_path = os.path.join(self.file_root, file_name)
            
            try:
                mat_data = scio.loadmat(file_path)
                # 兼容可能的 key: 'data' 或 'signal'
                raw_data = mat_data['data'] if 'data' in mat_data else mat_data['signal']
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return {}
            
            # 3. 数据预处理 (参考你的训练代码)
            # 确保形状是 [9, 2800] (Antenna, Time)
            # 如果原数据是 [2800, 9]，需要转置
            if raw_data.shape[0] > raw_data.shape[1]: 
                raw_data = raw_data.T 
                
            # 提取实部和虚部
            signal_real = torch.tensor(np.real(raw_data).astype(np.float32))
            signal_imag = torch.tensor(np.imag(raw_data).astype(np.float32))
            
            # 加入列表
            signals_list.append({
                'signal_real': signal_real,
                'signal_imag': signal_imag
            })

        # 4. 返回符合 Model 输入的格式
        return {
            "signals": signals_list,     # 这是一个包含4个字典的List ✅
            "filepath": self.view_files[0], # 用第一个文件名作为保存时的前缀
            "labels": 0                  # 占位符
        }

    def __len__(self):
        return 1 # 我们只推理这 1 组物体