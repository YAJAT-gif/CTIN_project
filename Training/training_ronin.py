import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from ctin_project.model.ctin_model import CTINModel
from ctin_project.loss.multitask_loss import VelocityFromPositionLoss

# === Config ===
csv_path = "output.csv"
window_size = 200
stride = 10
batch_size = 64
lr = 1e-4
epochs = 40
dt = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Custom Dataset ===
class IMUDataset(Dataset):
    def __init__(self, csv_path, window_size, stride):
        df = pd.read_csv(csv_path).dropna()
        self.imu = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
        self.pos = df[['gt_x', 'gt_y']].values

        # Normalize IMU
        self.imu_mean = self.imu.mean(axis=0)
        self.imu_std = self.imu.std(axis=0) + 1e-6
        self.imu = (self.imu - self.imu_mean) / self.imu_std

        # Create sliding windows
        self.X, self.Y = [], []
        for i in range(0, len(self.imu) - window_size + 1, stride):
            imu_win = self.imu[i:i+window_size]
            pos_win = self.pos[i:i+window_size]
            if imu_win.shape[0] == window_size:
                self.X.append(imu_win)
                self.Y.append(pos_win)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.Y[idx], dtype=torch.float32)
        )

# === Prepare Data ===
dataset = IMUDataset(csv_path, window_size, stride)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Model, Loss, Optimizer ===
model = CTINModel().to(device)
criterion = VelocityFromPositionLoss(dt=dt)
optimizer = optim.Adam(model.parameters(), lr=lr)

# === Training Loop ===
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for imu_batch, pos_batch in loader:
        imu_batch = imu_batch.to(device)      # [B, T, 6]
        pos_batch = pos_batch.to(device)      # [B, T, 2]

        optimizer.zero_grad()
        pred_vel, _ = model(imu_batch)        # [B, T, 2]
        loss = criterion(pred_vel, pos_batch) # compare with GT velocity
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

# === Save model ===
torch.save(model.state_dict(), "ctin_model_tlio_clean.pth")
print("Model saved to ctin_model_tlio_clean.pth")