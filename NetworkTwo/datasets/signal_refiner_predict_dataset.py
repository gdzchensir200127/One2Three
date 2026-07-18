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



class SignalRefinerPredictDataset(BaseDataset):

    def __init__(self, file_root, file_list_name):
        super().__init__()
        self.file_root = file_root

        # Read file list
        with open(os.path.join(self.file_root, "meta", file_list_name + ".txt"), "r") as fp:
            self.file_names = fp.read().split("\n")[:-1]

    def __getitem__(self, index):
        simulate_signal_path = self.file_names[index]
        simulate_signal = scio.loadmat(simulate_signal_path)['data']
        simulate_signal_I = np.real(simulate_signal).astype(np.float32)
        simulate_signal_I = torch.tensor(simulate_signal_I)
        simulate_signal_Q = np.imag(simulate_signal).astype(np.float32)
        simulate_signal_Q = torch.tensor(simulate_signal_Q)
        simulate_signal_phase = np.angle(simulate_signal).astype(np.float32)
        simulate_signal_phase = torch.tensor(simulate_signal_phase)
        simulate_signal_abs = torch.tensor(np.abs(simulate_signal).astype(np.float32))*100
        return {
            "simulate_signal_I": simulate_signal_I,
            "simulate_signal_Q": simulate_signal_Q,
            "simulate_signal_abs": simulate_signal_abs,
            "filepath": simulate_signal_path,
            "origin_phase": simulate_signal_phase
        }

    def __len__(self):
        return len(self.file_names)

