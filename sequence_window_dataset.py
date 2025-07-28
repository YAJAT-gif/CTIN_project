import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceWindowDataset(Dataset):
    def __init__(self, csv_dir, window_size=200, stride=1, normalize=True):
        self.window_size = window_size
        self.stride = stride
        self.samples = []

        for file in sorted(os.listdir(csv_dir)):
            if not file.endswith(".csv"):
                continue

            df = pd.read_csv(os.path.join(csv_dir, file)).dropna()
            imu = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
            vel = df[['vel_x', 'vel_y']].values

            # Normalize IMU per sequence (like TLIO)
            if normalize:
                imu_mean = imu.mean(axis=0)
                imu_std = imu.std(axis=0) + 1e-6
                imu = (imu - imu_mean) / imu_std

            # Clip extreme values if needed
            vel = np.clip(vel, -10, 10)

            # Windowing
            for i in range(0, len(imu) - window_size + 1, stride):
                x_window = imu[i:i+window_size]
                y_window = vel[i:i+window_size]
                if x_window.shape[0] == window_size and y_window.shape[0] == window_size:
                    self.samples.append((x_window, y_window))

        print(f" Total windows created: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
