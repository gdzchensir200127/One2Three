import json
import os
import pickle
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset
import trimesh

class ShapeNet_S2P_DualView(Dataset):
    """
    [Dual View Version]
    Loads View 0 (Front) and View 1 (Side), fills View 2, 3 with Zeros.
    """

    def __init__(self, file_root, file_list_name, mesh_pos, normalization, shapenet_options):
        super().__init__()
        self.file_root = file_root
        with open(os.path.join(self.file_root, "meta", shapenet_options.shapenet_json), "r") as fp:
            self.json = json.load(fp)
            self.labels_map = sorted(list(self.json.keys()))
        self.labels_map = {k: i for i, k in enumerate(self.labels_map)}
        
        with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")[:-1]
        self.file_names = [file_name for file_name in self.file_names if file_name.endswith('0.dat')]
        
        self.tensorflow = "_tf" in file_list_name
        self.normalization = normalization
        self.mesh_pos = mesh_pos
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border

    def __getitem__(self, index):
        if self.tensorflow:
            filename = self.file_names[index][17:]
            splits = filename.split("/")
            label = splits[0]
            object_id = splits[1]
            pkl_path = os.path.join(self.file_root, filename)
            
            view_index = int(pkl_path[-5])
            
            signals = []
            
            zeros_real = torch.zeros(9, 2800).float()
            zeros_imag = torch.zeros(9, 2800).float()
            
            # === 核心修改：指定保留正面(0) 和 侧面(1) ===
            target_view_indices = [0, 1] 

            # 用于暂存读取到的真实数据
            # key: view_id, value: {'signal_real': tensor, 'signal_imag': tensor}
            loaded_signals = {} 

            # 1. 先读取所需的真实数据
            for target_id in target_view_indices:
                # 构造对应视角的 .mat 文件路径
                # 逻辑：直接拼接目标视角ID，不再依赖循环变量 i
                signal_path = pkl_path[:-5] + str(target_id) + ".mat"
                
                try:
                    signal_data = scio.loadmat(signal_path)['data']
                    signal_real = torch.tensor(np.transpose(np.real(signal_data),(1,0)).astype(np.float32))
                    signal_imag = torch.tensor(np.transpose(np.imag(signal_data),(1,0)).astype(np.float32))
                    loaded_signals[target_id] = {'signal_real': signal_real, 'signal_imag': signal_imag}
                except Exception as e:
                    print(f"Error loading view {target_id} for {filename}: {e}")
                    # 容错：填全0
                    zeros_real = torch.zeros(9, 2800).float()
                    zeros_imag = torch.zeros(9, 2800).float()
                    loaded_signals[target_id] = {'signal_real': zeros_real, 'signal_imag': zeros_imag}

            # 2. 构造最终的 signals 列表 (重复填充)
            # 目标顺序：[View 0, View 1, View 0, View 1]
            signals = []
            
            # Slot 0: View 0
            signals.append(loaded_signals[0])
            
            # Slot 1: View 1
            signals.append(loaded_signals[1])
            
            # Slot 2: View 0 (重复)
            # 使用 .clone() 确保 tensor 内存独立，避免潜在的 inplace 操作 bug
            signals.append({
                'signal_real': loaded_signals[0]['signal_real'].clone(),
                'signal_imag': loaded_signals[0]['signal_imag'].clone()
            })
            
            # Slot 3: View 1 (重复)
            signals.append({
                'signal_real': loaded_signals[1]['signal_real'].clone(),
                'signal_imag': loaded_signals[1]['signal_imag'].clone()
            })
            
            # Ground Truth 加载部分保持不变
            project_path = os.path.join(self.file_root, label + "_depth_fixed4", object_id + ".mat")
            if not os.path.exists(project_path):
                splits = object_id.split("_")
                object_id = splits[0]
                project_path = os.path.join(self.file_root, label + "_depth_fixed4", object_id + ".mat")
            
            project = scio.loadmat(project_path)['Z']
            project = torch.tensor(project.astype(np.float32))
            save_path = os.path.join(self.file_root, 'output', label, object_id)
            
            pc_path = project_path.replace('.mat', '.ply')
            mesh = trimesh.load(pc_path)
            if isinstance(mesh, trimesh.PointCloud):
                points = mesh.vertices
            else:
                points = mesh.vertices
            points = torch.tensor(points.astype(np.float32))

        else:
            # Fallback for pickle format
            label, filename = self.dir_names[index].split("_", maxsplit=1)
            with open(os.path.join(self.file_root, "data", label, filename), "rb") as f:
                data = pickle.load(f, encoding="latin1")
            img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]
            points = pts

        return {
            "signals": signals, # 这里的 list 包含 [Real, Real, 0, 0]
            "points": points,
            "labels": self.labels_map.get(label, 0),
            "filename": filename,
            "project": project,
            "save_path": save_path
        }

    def __len__(self):
        return len(self.file_names)