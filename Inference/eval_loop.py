import torch
import numpy as np
import matplotlib.pyplot as plt
from model.ctin_model import CTINModel

# === Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalize_velocity = False
max_speed = 3.0
dt = 1 / 100

# === Load full ground truth velocity ===
vel_true = torch.load("Y_full_noisy.pt").numpy()
if normalize_velocity:
    vel_true = vel_true * max_speed

# === Load CTIN model and inputs ===
model = CTINModel().to(device)
model.load_state_dict(torch.load("ctin_synthetic_loop_noisy.pth", map_location=device))
model.eval()
X = torch.load("X_synthetic_noisy.pt")

# === Predict ===
with torch.no_grad():
    pred, _ = model(X.to(device))  # [N, 200, 2]
    if normalize_velocity:
        pred = pred * max_speed
    pred = pred.cpu().numpy()

# === Stitch predicted velocity: Overlap-aware reconstruction ===
# This averages overlapping predictions (better than just taking one per window)

window_size = X.shape[1]
num_windows = pred.shape[0]
total_length = num_windows + window_size - 1

vel_pred_full = np.zeros((total_length, 2))
counts = np.zeros((total_length, 1))

for i in range(num_windows):
    vel_pred_full[i:i+window_size] += pred[i]
    counts[i:i+window_size] += 1

vel_pred_full /= counts  # element-wise average

# === Integrate velocity to get position ===
pos_true = np.cumsum(vel_true * dt, axis=0)
pos_pred = np.cumsum(vel_pred_full * dt, axis=0)

# === Plot ===
plt.figure(figsize=(8, 8))
plt.plot(pos_true[:, 0], pos_true[:, 1], label="Ground Truth Path", linewidth=2)
plt.plot(pos_pred[:, 0], pos_pred[:, 1], label="Predicted Path", linestyle='--', linewidth=2)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("CTIN Full Trajectory (GT vs Predicted)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np

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
