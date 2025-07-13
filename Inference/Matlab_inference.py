import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from model.ctin_model import CTINModel
from loss.multitask_loss import MultiTaskLoss  # If needed for later eval

# ==== Config ====
window_size = 200
stride = 1
dt = 1 / 100  # 100 Hz
vel_scale = 40.0  # same as used in training

# ==== Load IMU and GT Velocity ====
imu = pd.read_csv("imu_ctin.csv", header=None).values.astype(np.float32)  # [T, 6]
vel_gt = pd.read_csv("gt_vel_ctin.csv", header=None).values.astype(np.float32)  # [T, 2]

# ==== Normalize IMU (use mean/std from training set ideally) ====
imu_mean = imu.mean(axis=0)
imu_std = imu.std(axis=0)
imu_norm = (imu - imu_mean) / (imu_std + 1e-6)

# ==== Create sliding windows ====
def create_windows(data, window_size, stride):
    return np.stack([data[i:i+window_size] for i in range(0, len(data)-window_size+1, stride)])

X_win = create_windows(imu_norm, window_size, stride=stride)  # [N, 200, 6]
X_tensor = torch.tensor(X_win, dtype=torch.float32)

# ==== Load Model ====
model = CTINModel()
model.load_state_dict(torch.load("ctin_synthetic_loop_noisy.pth", map_location='cpu'))
model.eval()

# ==== Predict ====
with torch.no_grad():
    pred_vel, _ = model(X_tensor)
    pred_vel = pred_vel * vel_scale  # denormalize
    pred_vel = pred_vel.numpy()      # [N, 200, 2]

# ==== Extract center-point velocity ====
center_idx = window_size // 2
center_vel = pred_vel[:, center_idx]  # [N, 2]

# ==== Interpolate to match full GT range ====
center_time = np.arange(len(center_vel)) + center_idx
t_full = np.arange(center_time[0], center_time[-1] + 1)

vx_interp = interp1d(center_time, center_vel[:, 0], kind='linear', fill_value="extrapolate")
vy_interp = interp1d(center_time, center_vel[:, 1], kind='linear', fill_value="extrapolate")
vel_pred_interp = np.stack([vx_interp(t_full), vy_interp(t_full)], axis=1)

# ==== Align GT velocity ====
vel_gt = vel_gt[center_time[0]:center_time[-1]+1]

# ==== Integrate to get position ====
pos_gt = np.cumsum(vel_gt * dt, axis=0)
pos_pred = np.cumsum(vel_pred_interp * dt, axis=0)

# ==== Error Metrics ====
def compute_ATE(pos_true, pos_pred):
    return np.sqrt(np.mean(np.sum((pos_true - pos_pred)**2, axis=1)))

def compute_T_RTE(pos_true, pos_pred, dt, interval_sec=2):
    ti = int(interval_sec / dt)
    errors = []
    for t in range(len(pos_true) - ti):
        gt_delta = pos_true[t + ti] - pos_true[t]
        pred_delta = pos_pred[t + ti] - pos_pred[t]
        errors.append(np.linalg.norm(gt_delta - pred_delta))
    return np.sqrt(np.mean(np.square(errors)))

def compute_D_RTE(pos_true, pos_pred, d=1.0):
    errors = []
    t = 0
    while t < len(pos_true) - 1:
        start = t
        for i in range(t+1, len(pos_true)):
            if np.linalg.norm(pos_true[i] - pos_true[start]) >= d:
                gt_delta = pos_true[i] - pos_true[start]
                pred_delta = pos_pred[i] - pos_pred[start]
                errors.append(np.linalg.norm(gt_delta - pred_delta))
                t = i
                break
        else:
            break
    return np.sqrt(np.mean(np.square(errors)))

def compute_PDE(pos_true, pos_pred):
    drift = np.linalg.norm(pos_true[-1] - pos_pred[-1])
    total_len = np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1))
    return drift / total_len

# ==== Evaluate ====
ate = compute_ATE(pos_gt, pos_pred)
t_rte = compute_T_RTE(pos_gt, pos_pred, dt=dt)
d_rte = compute_D_RTE(pos_gt, pos_pred)
pde = compute_PDE(pos_gt, pos_pred)

print(f"ATE:   {ate:.3f} m")
print(f"T-RTE: {t_rte:.3f} m")
print(f"D-RTE: {d_rte:.3f} m")
print(f"PDE:   {pde:.3f}")

# ==== Plot Trajectories ====
plt.figure(figsize=(8, 8))
plt.plot(pos_gt[:, 0], pos_gt[:, 1], label="Ground Truth", linewidth=2)
plt.plot(pos_pred[:, 0], pos_pred[:, 1], '--', label="Predicted (CTIN)", linewidth=2)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("CTIN Trajectory on MATLAB-Simulated UAV Path")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
