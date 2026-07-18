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



class SignalRefinerDataset(BaseDataset):

    def __init__(self, file_root, file_list_name):
        super().__init__()
        self.file_root = file_root

        # Read file list
        with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")[:-1]

    def __getitem__(self, index):
        simulate_signal_path = self.file_names[index]
        GT_signal_path = simulate_signal_path.replace('real_dataset_simulate_signal_aug', 'real_dataset_real_signal_aug')
        simulate_signal = scio.loadmat(simulate_signal_path)['data']
        simulate_signal_I = np.real(simulate_signal).astype(np.float32)
        simulate_signal_I = torch.tensor(simulate_signal_I)
        simulate_signal_Q = np.imag(simulate_signal).astype(np.float32)
        simulate_signal_Q = torch.tensor(simulate_signal_Q)
        simulate_signal_abs = torch.tensor(np.abs(simulate_signal).astype(np.float32))*100
        GT_signal = scio.loadmat(GT_signal_path)['data']

        GT_signal_I = np.real(GT_signal).astype(np.float32)
        GT_signal_I = torch.tensor(GT_signal_I)
        GT_signal_Q = np.imag(GT_signal).astype(np.float32)
        GT_signal_Q = torch.tensor(GT_signal_Q)
        GT_signal_abs = torch.tensor(np.abs(GT_signal).astype(np.float32))*100
        GT_signal_abs_max_value = torch.max(GT_signal_abs)
        GT_signal_abs_mask = (GT_signal_abs >= GT_signal_abs_max_value*0.1).float()
        return {
            "simulate_signal_I": simulate_signal_I,
            "simulate_signal_Q": simulate_signal_Q,
            "simulate_signal_abs": simulate_signal_abs,
            "GT_signal_I": GT_signal_I,
            "GT_signal_Q": GT_signal_Q,
            "GT_signal_abs": GT_signal_abs,
            "GT_signal_abs_mask" : GT_signal_abs_mask
        }

    def __len__(self):
        return len(self.file_names)

