import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from ctin_project.model.ctin_model import CTINModel
from ctin_project.loss.multitask_loss import MultiTaskLoss

import os
from ctin_project.utils.path_utils import get_dataset_path


imu_path = get_dataset_path("imu_ctin.csv")
print("Resolved IMU path:", imu_path)

assert os.path.exists(imu_path), "IMU CSV not found!"
# === Load synthetic loop dataset ===
# X_tensor = torch.load("X_circuit.pt")  # [N, 200, 6]
# Y_tensor = torch.load("Y_circuit.pt")  # [N, 200, 2]

import numpy as np
import pandas as pd

# === Load CSVs ===
imu_data = pd.read_csv("imu_ctin.csv", header=None).values  # [T, 6]
vel_data = pd.read_csv("gt_vel_ctin.csv", header=None).values  # [T, 2]


# Normalize IMU input (z-score)
# Normalize IMU
imu_mean = imu_data.mean(axis=0)
imu_std = imu_data.std(axis=0)
imu_data = (imu_data - imu_mean) / (imu_std + 1e-6)

# Normalize velocity
vel_scale = 40.0
vel_data = vel_data / vel_scale


print("Velocity max:", vel_data.max(axis=0))
print("Velocity min:", vel_data.min(axis=0))
print("Velocity std:", vel_data.std(axis=0))


# === Windowing function ===
def create_windows(data, targets, window_size=200, stride=1):
    X, Y = [], []
    for i in range(0, len(data) - window_size + 1, stride):
        if data[i:i+window_size].shape[0] != window_size:
            continue
        if targets[i:i+window_size].shape[0] != window_size:
            continue
        X.append(data[i:i+window_size])
        Y.append(targets[i:i+window_size])
    return np.stack(X), np.stack(Y)

# === Create windows ===
X_np, Y_np = create_windows(imu_data, vel_data, window_size=200, stride=1)
X_tensor = torch.tensor(X_np, dtype=torch.float32)
Y_tensor = torch.tensor(Y_np, dtype=torch.float32)

print("Loaded dataset:", X_tensor.shape, Y_tensor.shape)



# === Create dataset and loader ===
dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# === Model setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTINModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=8e-4)
criterion = MultiTaskLoss()

# === Training loop ===
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    total_loss, vel_loss, cov_loss = 0.0, 0.0, 0.0

    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        pred_vel, pred_cov = model(X_batch)  # velocity: [B, 200, 2], covariance: [B, 200, 2x2]
        loss, l_vel, l_cov = criterion(pred_vel, pred_cov, Y_batch, return_individual=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        vel_loss += l_vel
        cov_loss += l_cov

    avg_total = total_loss / len(train_loader)
    avg_vel = vel_loss / len(train_loader)
    avg_cov = cov_loss / len(train_loader)
    print(f"Epoch {epoch+1:2d} | Total: {avg_total:.4f} | Vel: {avg_vel:.4f} | Cov: {avg_cov:.4f}")

# === Save model for inference ===
torch.save(model.state_dict(), "ctin_synthetic_loop_noisy.pth")
print("Model saved as ctin_synthetic_loop_noisy.pth")
