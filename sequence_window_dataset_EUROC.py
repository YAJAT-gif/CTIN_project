import os, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class SequenceWindowDataset(Dataset):
    def __init__(self, csv_dir, window_size=200, stride=1, normalize=True,
                 std_floor=1e-3, imu_clip=20.0, z_clip=10.0, include_seq_id=False):
        """
        std_floor: minimum std used in z-scoring to avoid huge scales
        imu_clip: pre-normalization hard clip for raw IMU (m/s^2 and rad/s)
        z_clip: post-normalization clip to bound outliers
        """
        self.window_size = window_size
        self.stride = stride
        self.include_seq_id = include_seq_id
        self.samples = []

        csv_files = [f for f in sorted(os.listdir(csv_dir)) if f.endswith(".csv")]
        seq_counter = 0

        for file in csv_files:
            path = os.path.join(csv_dir, file)
            df = pd.read_csv(path)

            # Drop rows with NaNs in used columns only
            cols = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','vx','vy','vz']
            df = df[cols].dropna()

            imu = df[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']].to_numpy(dtype=np.float32)
            vel = df[['vx','vy','vz']].to_numpy(dtype=np.float32)

            # Optional raw IMU clamp (protect against spikes)
            if imu_clip is not None:
                np.clip(imu, -imu_clip, imu_clip, out=imu)


            if normalize:
                mean = imu.mean(axis=0, dtype=np.float64)
                std = imu.std(axis=0, dtype=np.float64)
                # Floor the std to avoid division by ~0; if truly constant, just center without scaling
                std = np.maximum(std, std_floor)
                imu = (imu - mean.astype(np.float32)) / std.astype(np.float32)
                if z_clip is not None:
                    np.clip(imu, -z_clip, z_clip, out=imu)


            np.clip(vel, -10.0, 10.0, out=vel)

            # Windowing strictly within this file/sequence
            n = len(imu)
            for i in range(0, n - window_size + 1, stride):
                xw = imu[i:i+window_size]
                yw = vel[i:i+window_size]
                if self.include_seq_id:
                    self.samples.append((xw, yw, seq_counter))
                else:
                    self.samples.append((xw, yw))
            seq_counter += 1

        print(f"Total windows created: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.include_seq_id:
            x, y, sid = self.samples[idx]
            return (torch.from_numpy(x), torch.from_numpy(y), torch.tensor(sid, dtype=torch.long))
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)
