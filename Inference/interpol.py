import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import numpy as np
import pandas as pd

# === Config ===
dt = 1 / 100  # 100 Hz
window_size = 200
normalize_velocity = True
max_speed = 40.0

# === Load CTIN model ===
from model.ctin_model import CTINModel  # adjust if needed
model = CTINModel()
model.load_state_dict(torch.load("ctin_synthetic_loop_noisy.pth", map_location='cpu'))  # adjust if needed
model.eval()

# === Load data ===
X = pd.read_csv("imu_ctin.csv", header=None).values  # [T, 6]
vel_true = pd.read_csv("gt_vel_ctin.csv", header=None).values  # [T, 2]
print("vel_true shape:", vel_true.shape)
if normalize_velocity:
    max_speed = 40.0
    vel_true *= max_speed

# === Run model to get center-point velocities ===
with torch.no_grad():
    pred, _ = model(X)  # [N, 200, 2]
    if normalize_velocity:
        pred = pred * max_speed
    pred = pred.numpy()

# === Extract center velocities
center_idx = window_size // 2
center_vel = pred[:, center_idx]  # [N, 2]

# === Interpolate to full resolution
T = len(vel_true)
center_time = np.arange(len(center_vel)) + center_idx

vx_interp = interp1d(center_time, center_vel[:, 0], kind='linear', bounds_error=False, fill_value="extrapolate")
vy_interp = interp1d(center_time, center_vel[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")

t_full = np.arange(T)
vel_interp = np.stack([vx_interp(t_full), vy_interp(t_full)], axis=1)  # [T, 2]

# === Integrate to position
pos_true = np.cumsum(vel_true * dt, axis=0)
pos_pred = np.cumsum(vel_interp * dt, axis=0)
print("GT vel:", vel_true.shape)
print("Interpolated vel:", vel_interp.shape)

def compute_ATE(pos_true, pos_pred):
    errors = pos_true - pos_pred
    squared_errors = np.sum(errors**2, axis=1)
    ate = np.sqrt(np.mean(squared_errors))
    return ate
import numpy as np

def compute_T_RTE(pos_true, pos_pred, dt, interval_sec=60):
    ti = int(interval_sec / dt)
    errors = []
    for t in range(len(pos_true) - ti):
        gt_delta = pos_true[t + ti] - pos_true[t]
        pred_delta = pos_pred[t + ti] - pos_pred[t]
        error = np.linalg.norm(gt_delta - pred_delta)
        errors.append(error)
    return np.sqrt(np.mean(np.square(errors)))

def compute_D_RTE(pos_true, pos_pred, d=1.0):
    errors = []
    gt_len = 0
    t = 0
    while t < len(pos_true) - 1:
        start = t
        for i in range(t + 1, len(pos_true)):
            dist = np.linalg.norm(pos_true[i] - pos_true[start])
            if dist >= d:
                gt_delta = pos_true[i] - pos_true[start]
                pred_delta = pos_pred[i] - pos_pred[start]
                error = np.linalg.norm(gt_delta - pred_delta)
                errors.append(error)
                t = i  # Move forward
                break
        else:
            break
    return np.sqrt(np.mean(np.square(errors)))

def compute_PDE(pos_true, pos_pred):
    drift = np.linalg.norm(pos_true[-1] - pos_pred[-1])
    total_length = np.sum(np.linalg.norm(np.diff(pos_true, axis=0), axis=1))
    return drift / total_length

# Call it:
ate = compute_ATE(pos_true, pos_pred)
t_rte = compute_T_RTE(pos_true, pos_pred, dt=1/100, interval_sec=2)  # use 2 sec for your synthetic loop
d_rte = compute_D_RTE(pos_true, pos_pred, d=1.0)
pde   = compute_PDE(pos_true, pos_pred)

print(f"T-RTE: {t_rte:.3f} m")
print(f"D-RTE: {d_rte:.3f} m")
print(f"PDE:   {pde:.3f}")

print(f"ATE: {ate:.3f} meters")


# === Plot
plt.figure(figsize=(8, 8))
plt.plot(pos_true[:, 0], pos_true[:, 1], label="Ground Truth", linewidth=2)
plt.plot(pos_pred[:, 0], pos_pred[:, 1], '--', label="Predicted (Center + Interp)", linewidth=2)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("CTIN Trajectory: Center-Point Interpolation")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
