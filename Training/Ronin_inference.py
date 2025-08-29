import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ctin_project.model.ctin_model import CTINModel

# === Config ===
csv_path = "output.csv"
model_path = "ctin_model_tlio_clean.pth"
window_size = 200
stride = 10
dt = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load and normalize IMU ===
df = pd.read_csv(csv_path).dropna()
imu = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
gt_pos = df[['gt_x', 'gt_y']].values

imu_mean = imu.mean(axis=0)
imu_std = imu.std(axis=0) + 1e-6
imu = (imu - imu_mean) / imu_std

# === Create windows ===
def create_windows(data, window_size, stride):
    return np.stack([
        data[i:i+window_size]
        for i in range(0, len(data) - window_size + 1, stride)
        if data[i:i+window_size].shape[0] == window_size
    ])

X = create_windows(imu, window_size, stride)
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# === Load model ===
model = CTINModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Predict center-frame velocities ===
center_vels = []
with torch.no_grad():
    for i in range(0, len(X_tensor), 64):
        batch = X_tensor[i:i+64]
        pred_vel, _ = model(batch)
        center_vel = pred_vel[:, window_size // 2, :]  # [B, 2]
        center_vels.append(center_vel.cpu())

center_vels = torch.cat(center_vels, dim=0)  # [N, 2]

# === Integrate to position ===
def integrate_velocity(vel, dt):
    pos = torch.zeros_like(vel)
    pos[0] = vel[0] * dt
    for t in range(1, len(vel)):
        pos[t] = pos[t-1] + vel[t-1] * dt
    return pos

pos_pred = integrate_velocity(center_vels, dt).numpy()

# === Align GT ===
start = window_size // 2
gt_aligned = gt_pos[start:start + stride * len(center_vels):stride]
gt_aligned = gt_aligned[:len(pos_pred)]  # just in case

# === Error Metrics ===
def compute_ATE(gt, pred):
    return np.mean(np.linalg.norm(gt - pred, axis=1))

def compute_PDE(gt, pred):
    drift = np.linalg.norm(gt[-1] - pred[-1])
    total = np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1))
    return drift / total

ate = compute_ATE(gt_aligned, pos_pred)
pde = compute_PDE(gt_aligned, pos_pred)

print(f"ATE: {ate:.4f} m")
print(f"PDE: {pde:.4f}")

# === Plot ===
plt.figure(figsize=(8, 6))
plt.plot(gt_aligned[:, 0], gt_aligned[:, 1], label="GT")
plt.plot(pos_pred[:, 0], pos_pred[:, 1], '--', label="CTIN Predicted")
plt.legend()
plt.axis("equal")
plt.title("CTIN Trajectory (TLIO-style Inference)")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.grid(True)
plt.tight_layout()
plt.show()
