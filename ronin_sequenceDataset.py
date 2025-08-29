import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class RoNINSequenceDataset(Dataset):
    def __init__(self, csv_path, window_size=200, stride=20, normalize_imu=True):
        self.samples = []
        df = pd.read_csv(csv_path).dropna()

        imu = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
        pos = df[['gt_x', 'gt_y']].values

        if normalize_imu:
            imu_mean = imu.mean(axis=0)
            imu_std = imu.std(axis=0) + 1e-6
            imu = (imu - imu_mean) / imu_std

        for i in range(0, len(imu) - window_size + 1, stride):
            x_window = imu[i:i + window_size]
            y_window = pos[i:i + window_size] - pos[i]
            anchor = pos[i]  # NEW: store GT anchor
            if x_window.shape[0] == window_size:
                self.samples.append((x_window, y_window, anchor))

        print(f"✅ Loaded {len(self.samples)} windows from {csv_path}")
        print(pos)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, anchor = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(anchor, dtype=torch.float32)
        )
