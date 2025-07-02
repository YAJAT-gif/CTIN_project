import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model.ctin_model import CTINModel
from loss.multitask_loss import MultiTaskLoss

# Dummy dataset
class IMUDataset(Dataset):
    def __init__(self, B=4, T=200, D=6):
        self.X = torch.randn(B, T, D)
        self.vel = torch.randn(B, T, 2)
        self.pos = torch.zeros(B, T, 2)
        self.pos[:, 0] = self.vel[:, 0] * 0.01
        for t in range(1, T):
            self.pos[:, t] = self.pos[:, t - 1] + self.vel[:, t - 1] * 0.01

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.vel[idx], self.pos[idx]

# Load model + data
dataset = IMUDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = CTINModel()
loss_fn = MultiTaskLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train for one batch
model.train()
for imu, vel_gt, pos_gt in loader:
    optimizer.zero_grad()
    vel_pred, cov_pred = model(imu)
    loss, lv, lc = loss_fn(vel_pred, cov_pred, vel_gt, pos_gt)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.4f} | Vel Loss: {lv:.4f} | Cov Loss: {lc:.4f}")
    break
