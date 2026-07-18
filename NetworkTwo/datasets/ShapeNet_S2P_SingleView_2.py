import json
import os
import pickle
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset
import trimesh

class ShapeNet_S2P_SingleView_2(Dataset):
    """
    [Single View Version - Repetition Mode]
    Input:  4 copies of View 1 (Side) Signal.
    Target: 4 copies of View 1 (Side) Depth Map.
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
            
            # === 1. 信号读取改动 (重复输入) ===
            target_view_idx = 1 # 指定只用侧面 (Index 1)
            
            # 构造侧面信号文件的绝对路径
            # 原文件名逻辑假设: path/.../model_0.mat, model_1.mat ...
            side_signal_path = pkl_path[:-5] + str(target_view_idx) + ".mat"
            
            try:
                signal_data = scio.loadmat(side_signal_path)['data']
                side_real = torch.tensor(np.transpose(np.real(signal_data),(1,0)).astype(np.float32))
                side_imag = torch.tensor(np.transpose(np.imag(signal_data),(1,0)).astype(np.float32))
            except Exception as e:
                print(f"Error loading signal {side_signal_path}: {e}")
                side_real = torch.zeros(9, 2800).float()
                side_imag = torch.zeros(9, 2800).float()

            # 构造 signals 列表：复制 4 份侧面信号
            signals = []
            for _ in range(4):
                signals.append({
                    'signal_real': side_real.clone(),
                    'signal_imag': side_imag.clone()
                })
            
            # === 2. Ground Truth (Project) 读取改动 (重复 GT) ===
            project_path = os.path.join(self.file_root, label + "_depth_fixed4", object_id + ".mat")
            if not os.path.exists(project_path):
                splits = object_id.split("_")
                object_id = splits[0]
                project_path = os.path.join(self.file_root, label + "_depth_fixed4", object_id + ".mat")
            
            # 读取包含所有4个视角深度的矩阵
            # 假设 raw_project 的形状是 [4, 128, 128] (视具体数据集格式而定，通常是 [View, H, W])
            raw_project = scio.loadmat(project_path)['Z']
            
            # 提取侧面深度图 (Index 1)
            # 注意：这里假设 'Z' 的第一个维度是视角。如果报错，可能是 [H, W, 4]，需要改成 [:, :, target_view_idx]
            side_view_depth = raw_project[target_view_idx] 
            
            # 将侧面深度图复制 4 份，堆叠成 [4, 128, 128]
            # 这样网络的 4 个输出通道都会被监督去预测侧面
            project_repeated = np.stack([side_view_depth] * 4, axis=0)
            
            project = torch.tensor(project_repeated.astype(np.float32))
            
            save_path = os.path.join(self.file_root, 'output', label, object_id)
            
            # === 3. Point Cloud 读取 (保持不变) ===
            # 注意：这是完整的 3D 点云。计算 CD Loss 时，你的“侧面预测”和“完整点云”差距会很大
            # 但这是物理事实，不需要修改读取逻辑。
            pc_path = project_path.replace('.mat', '.ply')
            mesh = trimesh.load(pc_path)
            if isinstance(mesh, trimesh.PointCloud):
                points = mesh.vertices
            else:
                points = mesh.vertices
            points = torch.tensor(points.astype(np.float32))

        else:
            # Fallback (旧逻辑，未做单视角适配，建议忽略)
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
            "project": project, # 现在的 project 是 4 个一模一样的侧面深度图
            "save_path": save_path
        }

    def __len__(self):
        return len(self.file_names)