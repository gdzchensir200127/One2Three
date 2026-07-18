import json
import os
import pickle
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset
import trimesh

class ShapeNet_S2P_SingleView(Dataset):
    """
    [Single View Version]
    Only loads View 1 (Side), fills View 0, 2, 3 with Zeros.
    """

    def __init__(self, file_root, file_list_name, mesh_pos, normalization, shapenet_options):
        super().__init__()
        self.file_root = file_root
        # 读取元数据 JSON
        with open(os.path.join(self.file_root, "meta", shapenet_options.shapenet_json), "r") as fp:
            self.json = json.load(fp)
            self.labels_map = sorted(list(self.json.keys()))
        self.labels_map = {k: i for i, k in enumerate(self.labels_map)}
        
        # 读取文件列表
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
            
            # 获取当前文件的基础视角索引 (通常为0, 因为只筛选了0.dat)
            view_index = int(pkl_path[-5])
            
            signals = []
            
        
            zeros_real = torch.zeros(9, 2800).float()
            zeros_imag = torch.zeros(9, 2800).float()
            
            # === 核心修改：指定只保留侧面 (Index 1) ===
            target_view_idx = 1 

            side_signal_path = pkl_path[:-5] + str(target_view_idx) + ".mat"
            
            try:
                signal_data = scio.loadmat(side_signal_path)['data']
                # 转置处理
                side_real = torch.tensor(np.transpose(np.real(signal_data),(1,0)).astype(np.float32))
                side_imag = torch.tensor(np.transpose(np.imag(signal_data),(1,0)).astype(np.float32))
            except Exception as e:
                print(f"Error loading side view {side_signal_path}: {e}")
                # 容错：如果侧面文件都读不到，就填全0
                side_real = torch.zeros(9, 2800).float()
                side_imag = torch.zeros(9, 2800).float()

            # 3. 构造 signals 列表：放入 4 个一模一样的侧面信号
            signals = []
            for _ in range(4):
                # 使用 clone() 确保内存独立，虽然对于只读数据不 clone 也行，但 clone 更安全
                signals.append({
                    'signal_real': side_real.clone(), 
                    'signal_imag': side_imag.clone()
                })
            
            # Ground Truth 加载部分保持不变 (为了计算 Loss)
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
            # 兼容非TF格式的旧代码 (如果不需要可以忽略)
            label, filename = self.dir_names[index].split("_", maxsplit=1)
            with open(os.path.join(self.file_root, "data", label, filename), "rb") as f:
                data = pickle.load(f, encoding="latin1")
            img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]
            points = pts # 简单适配

        return {
            "signals": signals, # 这里的 list 包含 [0, Real, 0, 0]
            "points": points,
            "labels": self.labels_map.get(label, 0),
            "filename": filename,
            "project": project,
            "save_path": save_path
        }

    def __len__(self):
        return len(self.file_names)