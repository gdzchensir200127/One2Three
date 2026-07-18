import json
import os
import pickle
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset
import trimesh

class ShapeNet_S2P_DualView_2(Dataset):
    """
    [Dual View Version - Repetition Mode]
    Input:  [View 0, View 1, View 0, View 1] (Front, Side, Front, Side)
    Target: [View 0, View 1, View 0, View 1] (Depth Maps)
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
            
            # view_index 通常用于确定文件名的起始偏移，这里我们直接指定读取 0 和 1
            # 假设文件名格式为 ..._0.mat, ..._1.mat 等
            
            # === 1. 信号读取改动 (读取 0, 1 并重复) ===
            target_indices = [0, 2] # 正面(0) 和 侧面(1)
            loaded_signals = {} # 暂存读取到的数据

            # 先读取真实的 0 和 1
            for view_id in target_indices:
                signal_path = pkl_path[:-5] + str(view_id) + ".mat"
                try:
                    signal_data = scio.loadmat(signal_path)['data']
                    sig_real = torch.tensor(np.transpose(np.real(signal_data),(1,0)).astype(np.float32))
                    sig_imag = torch.tensor(np.transpose(np.imag(signal_data),(1,0)).astype(np.float32))
                    loaded_signals[view_id] = {'real': sig_real, 'imag': sig_imag}
                except Exception as e:
                    print(f"Error loading signal {signal_path}: {e}")
                    # 容错填0
                    loaded_signals[view_id] = {
                        'real': torch.zeros(9, 2800).float(),
                        'imag': torch.zeros(9, 2800).float()
                    }

            # 构造 signals 列表，顺序：[View0, View1, View0, View1]
            signals = []
            # 第1遍
            signals.append({'signal_real': loaded_signals[0]['real'], 'signal_imag': loaded_signals[0]['imag']})
            signals.append({'signal_real': loaded_signals[2]['real'], 'signal_imag': loaded_signals[2]['imag']})
            # 第2遍 (重复，使用 clone 确保安全)
            signals.append({'signal_real': loaded_signals[0]['real'].clone(), 'signal_imag': loaded_signals[0]['imag'].clone()})
            signals.append({'signal_real': loaded_signals[2]['real'].clone(), 'signal_imag': loaded_signals[2]['imag'].clone()})
            
            
            # === 2. Ground Truth (Project) 读取改动 (GT 也要重复) ===
            project_path = os.path.join(self.file_root, label + "_depth_fixed4", object_id + ".mat")
            if not os.path.exists(project_path):
                splits = object_id.split("_")
                object_id = splits[0]
                project_path = os.path.join(self.file_root, label + "_depth_fixed4", object_id + ".mat")
            
            # 读取包含4个视角的原始深度数据
            raw_project = scio.loadmat(project_path)['Z'] # 假设形状 [4, 128, 128]
            
            # 提取正面(0) 和 侧面(1)
            depth_0 = raw_project[0]
            depth_1 = raw_project[2]
            
            # 按照 [View0, View1, View0, View1] 的顺序堆叠
            # 这样 Loss 计算时：
            # Output_0 vs GT_0 (Front)
            # Output_1 vs GT_1 (Side)
            # Output_2 vs GT_0 (Front)
            # Output_3 vs GT_1 (Side)
            project_repeated = np.stack([depth_0, depth_1, depth_0, depth_1], axis=0)
            
            project = torch.tensor(project_repeated.astype(np.float32))
            
            save_path = os.path.join(self.file_root, 'output', label, object_id)
            
            # === 3. Point Cloud (保持不变) ===
            # 3D 点云依然是完整的物体，用于评估整体几何形状
            pc_path = project_path.replace('.mat', '.ply')
            mesh = trimesh.load(pc_path)
            if isinstance(mesh, trimesh.PointCloud):
                points = mesh.vertices
            else:
                points = mesh.vertices
            points = torch.tensor(points.astype(np.float32))

        else:
            # Fallback (旧逻辑，忽略)
            label, filename = self.dir_names[index].split("_", maxsplit=1)
            with open(os.path.join(self.file_root, "data", label, filename), "rb") as f:
                data = pickle.load(f, encoding="latin1")
            img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]
            points = pts

        return {
            "signals": signals,
            "points": points,
            "labels": self.labels_map.get(label, 0),
            "filename": filename,
            "project": project, # 这里的 project 已经是 [Front, Side, Front, Side]
            "save_path": save_path
        }

    def __len__(self):
        return len(self.file_names)