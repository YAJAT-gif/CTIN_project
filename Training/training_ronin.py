import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from ctin_project.model.ctin_model import CTINModel
from ctin_project.loss.multitask_loss import MultiTaskLoss
from ctin_project.utils.path_utils import get_dataset_path

# === Config ===
ronin_csv = "output.csv"
window_size = 200
stride = 15
batch_size = 16
num_epochs = 25
lr = 8e-4
vel_scale = 5.0  # Adjust based on magnitude in RoNIN data

# === Load RoNIN CSV ===
df = pd.read_csv(ronin_csv)

# === Extract IMU and GT Position ===
imu_data = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values  # [T, 6]
gt_pos = df[['gt_x', 'gt_y']].values  # [T, 2]
timestamps = df['timestamp'].values

# === Compute GT Velocity via finite difference ===
dt = np.gradient(timestamps)
vel_data = np.gradient(gt_pos, axis=0) / dt[:, None]  # [T, 2]

# === Optional: Smooth velocity to reduce GT noise ===
vel_data = gaussian_filter1d(vel_data, sigma=2, axis=0)

# === Normalize IMU ===
imu_mean = imu_data.mean(axis=0)
imu_std = imu_data.std(axis=0)
imu_data = (imu_data - imu_mean) / (imu_std + 1e-6)

# === Normalize Velocity ===
vel_data = vel_data / vel_scale

print("Velocity max:", vel_data.max(axis=0))
print("Velocity min:", vel_data.min(axis=0))
print("Velocity std:", vel_data.std(axis=0))

# === Create windowed sequences ===
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

X_np, Y_np = create_windows(imu_data, vel_data, window_size=window_size, stride=stride)
X_tensor = torch.tensor(X_np, dtype=torch.float32)
Y_tensor = torch.tensor(Y_np, dtype=torch.float32)

print("Loaded dataset:", X_tensor.shape, Y_tensor.shape)

# === Dataset and DataLoader ===
dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Model, Loss, Optimizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTINModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = MultiTaskLoss()

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    total_loss, vel_loss, cov_loss = 0.0, 0.0, 0.0

    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        pred_vel, pred_cov = model(X_batch)
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

# === Save Model ===
torch.save(model.state_dict(), "ctin_trained_on_ronin.pth")
print("✅ Model saved as ctin_trained_on_ronin.pth")
