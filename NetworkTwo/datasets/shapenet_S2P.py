import json
import os
import pickle

import numpy as np
import scipy.io as scio
import torch
from PIL import Image
from skimage import io, transform
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt

import config
from datasets.base_dataset import BaseDataset



class ShapeNet_S2P(BaseDataset):
    """
    Dataset wrapping signal and target meshes for ShapeNet dataset.
    """

    def __init__(self, file_root, file_list_name, mesh_pos, normalization, shapenet_options):
        super().__init__()
        self.file_root = file_root
        with open(os.path.join(self.file_root, "meta", "shapenet.json"), "r") as fp:
            self.labels_map = sorted(list(json.load(fp).keys()))
        self.labels_map = {k: i for i, k in enumerate(self.labels_map)}
        # Read file list
        with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")[:-1]
        self.tensorflow = "_tf" in file_list_name # tensorflow version of data
        self.normalization = normalization
        self.mesh_pos = mesh_pos
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border
        self.p2s = shapenet_options.p2s

    def __getitem__(self, index):
        if self.tensorflow:
            filename = self.file_names[index][17:]
            label = filename.split("/", maxsplit=1)[0]
            pkl_path = os.path.join(self.file_root, "data_tf", filename)
            signal_path = pkl_path[:-4] + ".mat"
            json_path = pkl_path[:-4] + ".json"
            with open(pkl_path) as f:
                data = pickle.load(open(pkl_path, 'rb'), encoding="latin1").astype(np.float32)
            with open(json_path, 'r') as f:
                js = json.load(f)
                start_bin = js['start_bin']
                end_bin = js['end_bin']

            pts, normals = data[:, :3], data[:, 3:]

            signal = scio.loadmat(signal_path)['data'] # 2800 * 9
            # if self.p2s:
            #     signal = signal[::10,:] # 280 * 9 下采样
            signal_real = torch.tensor(np.transpose(np.real(signal),(1,0)).astype(np.float32))
            signal_imag = torch.tensor(np.transpose(np.imag(signal),(1,0)).astype(np.float32))


        else:
            label, filename = self.file_names[index].split("_", maxsplit=1)
            with open(os.path.join(self.file_root, "data", label, filename), "rb") as f:
                data = pickle.load(f, encoding="latin1")
            img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]
        # if self.p2s:
        #     pts = torch.tensor(np.transpose(pts,(1,0)).astype(np.float32))
        return {
            "signal_real": signal_real,
            "signal_imag": signal_imag,
            "points": pts,
            "normals": normals,
            "labels": self.labels_map[label],
            "filename": filename,
            "length": length,
            "start_bin": start_bin,
            "end_bin": end_bin
        }

    def __len__(self):
        return len(self.file_names)

