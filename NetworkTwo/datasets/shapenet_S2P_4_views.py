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



class ShapeNet_S2P_4_views(BaseDataset):
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
            self.dir_names = fp.read().split("\n")[:-1]
        self.tensorflow = "_tf" in file_list_name # tensorflow version of data
        self.normalization = normalization
        self.mesh_pos = mesh_pos
        self.resize_with_constant_border = shapenet_options.resize_with_constant_border

    def __getitem__(self, index):
        if self.tensorflow:
            dirname = self.dir_names[index][17:]
            label = dirname.split("/", maxsplit=1)[0]
            pkl_path = os.path.join(self.file_root, "data_tf", dirname, '0.dat')
            with open(pkl_path) as f:
                data = pickle.load(open(pkl_path, 'rb'), encoding="latin1").astype(np.float32)
            pts, normals = data[:, :3], data[:, 3:]
            signals = []
            for i in range(4):
                signal_path = pkl_path[:-5] + str(i) + ".mat"
                signal = scio.loadmat(signal_path)['data']
                signal_real = torch.tensor(np.transpose(np.real(signal),(1,0)).astype(np.float32))
                signal_imag = torch.tensor(np.transpose(np.imag(signal),(1,0)).astype(np.float32))
                signals.append({'signal_real':signal_real,'signal_imag':signal_imag})


        else:
            label, filename = self.dir_names[index].split("_", maxsplit=1)
            with open(os.path.join(self.file_root, "data", label, filename), "rb") as f:
                data = pickle.load(f, encoding="latin1")
            img, pts, normals = data[0].astype(np.float32) / 255.0, data[1][:, :3], data[1][:, 3:]

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        return {
            "signals": signals,
            "points": pts,
            "normals": normals,
            "labels": self.labels_map[label],
            "dirname": dirname,
            "length": length
        }

    def __len__(self):
        return len(self.dir_names)

