import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import pandas as pd
from ctin_project.model.ctin_model import CTINModel
from ctin_project.loss.multitask_loss import MultiTaskLoss

# === Config ===
ronin_csv = "output.csv"
window_size = 200
stride = 10
batch_size = 64
num_epochs = 60
lr = 1e-4

# === Load RoNIN CSV ===
df = pd.read_csv(ronin_csv)
imu_data = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
gt_pos = df[['gt_x', 'gt_y']].values
timestamps = df['timestamp'].values

# === Normalize IMU globally ===
imu_mean = imu_data.mean(axis=0)
imu_std = imu_data.std(axis=0)
imu_data = (imu_data - imu_mean) / (imu_std + 1e-6)

# === Windowing with per-window normalization ===
def create_windows(data, targets, window_size=200, stride=1):
    X, Y = [], []
    for i in range(0, len(data) - window_size + 1, stride):
        imu_window = data[i:i+window_size]
        pos_window = targets[i:i+window_size]

        if imu_window.shape[0] != window_size or pos_window.shape[0] != window_size:
            continue

        # Normalize positions per window
        origin = pos_window[0]
        std = pos_window.std(axis=0) + 1e-6
        pos_window = (pos_window - origin) / std

        X.append(imu_window)
        Y.append(pos_window)
    return np.stack(X), np.stack(Y)

X_np, Y_np = create_windows(imu_data, gt_pos, window_size=window_size, stride=stride)
X_tensor = torch.tensor(X_np, dtype=torch.float32)
Y_tensor = torch.tensor(Y_np, dtype=torch.float32)

print("Loaded dataset:", X_tensor.shape, Y_tensor.shape)

# === Dataset and DataLoader ===
dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Model, Loss, Optimizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTINModel(output_mode="pos").to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = MultiTaskLoss()  # position loss + covariance

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    total_loss, pos_loss, cov_loss = 0.0, 0.0, 0.0

    for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        pred_pos, pred_cov = model(X_batch)
        loss, l_pos, l_cov = criterion(pred_pos, pred_cov, Y_batch, return_individual=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pos_loss += l_pos
        cov_loss += l_cov

        if epoch == 0 and batch_idx == 0:
            print("Sample pred:", pred_pos[0, :5].detach().cpu().numpy())
            print("Sample gt:  ", Y_batch[0, :5].detach().cpu().numpy())

    avg_total = total_loss / len(train_loader)
    avg_pos = pos_loss / len(train_loader)
    avg_cov = cov_loss / len(train_loader)
    print(f"Epoch {epoch+1:2d} | Total: {avg_total:.4f} | Pos: {avg_pos:.4f} | Cov: {avg_cov:.4f}")

# === Save Model ===
torch.save(model.state_dict(), "ctin_trained_on_ronin_pos.pth")
print("✅ Model saved as ctin_trained_on_ronin_pos.pth")
