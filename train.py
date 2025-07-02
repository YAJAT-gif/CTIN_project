import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
from model.ctin_model import CTINModel
from loss.multitask_loss import MultiTaskLoss
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Load and preprocess data
imu_df = pd.read_csv('imu_data.csv')
gt_df = pd.read_csv('gt_data.csv')

imu_df.rename(columns={"#timestamp [ns]": "timestamp"}, inplace=True)
gt_df.rename(columns={"#timestamp": "timestamp"}, inplace=True)

merged_df = pd.merge(imu_df, gt_df, on="timestamp", how="inner")
merged_df.columns = merged_df.columns.str.strip()

imu_data = merged_df[[
    "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]",
    "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"
]].values
gt_velocity = merged_df[["v_RS_R_x [m s^-1]", "v_RS_R_y [m s^-1]"]].values

# Rotate IMU to navigation frame
quaternions = merged_df[["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"]].values
rotations = R.from_quat(quaternions)
gyro_nav = rotations.apply(imu_data[:, :3])
accel_nav = rotations.apply(imu_data[:, 3:])
imu_nav_data = np.concatenate([gyro_nav, accel_nav], axis=1)

# Create sliding windows
window_size = 200
X_windows, Y_windows = [], []
for i in range(len(imu_nav_data) - window_size + 1):
    X_windows.append(imu_nav_data[i:i+window_size])
    Y_windows.append(gt_velocity[i:i+window_size])  # full sequence of 200 velocities


X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)
Y_tensor = torch.tensor(np.array(Y_windows), dtype=torch.float32)

X_tensor = X_tensor[:200]
Y_tensor = Y_tensor[:200]


# Dataset and loaders
full_dataset = TensorDataset(X_tensor, Y_tensor)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Model and training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTINModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = MultiTaskLoss()

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss, vel_loss, cov_loss = 0, 0, 0
    for X_batch, Y_batch in train_loader:
        # print("Batch shape:", X_batch.shape, Y_batch.shape)
        # break  # just to test one batch

        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        pred_vel, pred_cov = model(X_batch)
        # print("vel_pred:", pred_vel.shape)
        # print("Y_batch:", Y_batch.shape)

        loss, l_vel, l_cov = criterion(pred_vel, pred_cov, Y_batch, return_individual=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        vel_loss += l_vel
        cov_loss += l_cov

    avg_total = total_loss / len(train_loader)
    avg_vel = vel_loss / len(train_loader)
    avg_cov = cov_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Loss: {avg_total:.4f} | Vel Loss: {avg_vel:.4f} | Cov Loss: {avg_cov:.4f}")

# Save the trained model
torch.save(model.state_dict(), "ctin_model.pth")
