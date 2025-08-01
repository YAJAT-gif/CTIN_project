import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ctin_project.model.ctin_model import CTINModel

# === Config ===
ronin_csv = "output.csv"
model_path = "ctin_trained_on_ronin_pos.pth"
window_size = 200
stride = 10

# === Load data ===
df = pd.read_csv(ronin_csv)
imu_data = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
gt_pos = df[['gt_x', 'gt_y']].values
timestamps = df['timestamp'].values

# === Normalize IMU ===
imu_mean = imu_data.mean(axis=0)
imu_std = imu_data.std(axis=0)
imu_data = (imu_data - imu_mean) / (imu_std + 1e-6)

# === Create normalized windows and save stats ===
def create_windows(data, targets, window_size, stride):
    X, origins, stds, indices = [], [], [], []
    for i in range(0, len(data) - window_size + 1, stride):
        imu_window = data[i:i+window_size]
        pos_window = targets[i:i+window_size]

        if imu_window.shape[0] != window_size or pos_window.shape[0] != window_size:
            continue

        origin = pos_window[0]
        std = pos_window.std(axis=0) + 1e-6
        norm_pos = (pos_window - origin) / std

        X.append(imu_window)
        origins.append(origin)
        stds.append(std)
        indices.append(i)
    return np.stack(X), np.array(origins), np.array(stds), np.array(indices)

X_np, origins, stds, start_indices = create_windows(imu_data, gt_pos, window_size, stride)
X_tensor = torch.tensor(X_np, dtype=torch.float32)

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTINModel(output_mode="pos").to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Predict ===
all_pred = []
with torch.no_grad():
    for i in range(0, len(X_tensor), 32):
        X_batch = X_tensor[i:i+32].to(device)
        pred_pos, _ = model(X_batch)
        all_pred.append(pred_pos.cpu().numpy())

pred_pos_windows = np.concatenate(all_pred, axis=0)  # shape [N, 200, 2]

# === Denormalize each predicted window ===
denorm_windows = []
for i in range(len(pred_pos_windows)):
    pred = pred_pos_windows[i]
    std = stds[i]
    origin = origins[i]
    denorm = pred * std + origin
    denorm_windows.append(denorm)
pred_pos_windows = np.stack(denorm_windows)  # [N, 200, 2]

# === Reconstruct full trajectory via overlap averaging ===
full_pred_pos = np.zeros((len(imu_data), 2))
counts = np.zeros(len(imu_data))

for i, start in enumerate(start_indices):
    full_pred_pos[start:start+window_size] += pred_pos_windows[i]
    counts[start:start+window_size] += 1

counts[counts == 0] = 1
pred_pos = full_pred_pos / counts[:, None]

# === Metrics ===
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
                pred_delta = pos_pred[i] - pred_pos[start]
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

dt = float(np.mean(np.diff(timestamps)))
ate = compute_ATE(gt_pos, pred_pos)
t_rte = compute_T_RTE(gt_pos, pred_pos, dt=dt)
d_rte = compute_D_RTE(gt_pos, pred_pos)
pde = compute_PDE(gt_pos, pred_pos)

print(f"ATE:   {ate:.3f} m")
print(f"T-RTE: {t_rte:.3f} m")
print(f"D-RTE: {d_rte:.3f} m")
print(f"PDE:   {pde:.3f}")

# === Plot ===
plt.figure(figsize=(10, 6))
plt.plot(gt_pos[:, 0], gt_pos[:, 1], label='Ground Truth',linewidth=4)
plt.plot(pred_pos[:, 0], pred_pos[:, 1], label='CTIN Prediction',linestyle='--')
plt.title("CTIN Trajectory vs Ground Truth (Position Prediction)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
