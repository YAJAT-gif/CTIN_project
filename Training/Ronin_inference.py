import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ctin_project.model.ctin_model import CTINModel
from ctin_project.utils.path_utils import get_dataset_path

# === Config (match training settings) ===
ronin_csv = "output.csv"
model_path = "ctin_trained_on_ronin.pth"
window_size = 200
stride = 20
vel_scale = 1.0

# === Load data ===
df = pd.read_csv(ronin_csv)
imu_data = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
gt_pos = df[['gt_x', 'gt_y']].values
timestamps = df['timestamp'].values

# === Normalize IMU (use stats from inference dataset) ===
imu_mean = imu_data.mean(axis=0)
imu_std = imu_data.std(axis=0)
imu_data = (imu_data - imu_mean) / (imu_std + 1e-6)

# === Windowing ===
def create_windows(data, window_size, stride):
    X, indices = [], []
    for i in range(0, len(data) - window_size + 1, stride):
        X.append(data[i:i+window_size])
        indices.append(i)
    return np.stack(X), np.array(indices)

X_np, start_indices = create_windows(imu_data, window_size, stride)
X_tensor = torch.tensor(X_np, dtype=torch.float32)

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTINModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Predict velocity windows ===
all_pred_vel = []

with torch.no_grad():
    for i in range(0, len(X_tensor), 32):  # small batch size to avoid OOM
        X_batch = X_tensor[i:i+32].to(device)
        pred_vel, _ = model(X_batch)
        all_pred_vel.append(pred_vel.cpu().numpy())

pred_vel_windows = np.concatenate(all_pred_vel, axis=0)  # shape: [N, 200, 2]
pred_vel_windows = pred_vel_windows * vel_scale  # denormalize

# === Reconstruct full-resolution velocity via overlap averaging ===
full_pred_vel = np.zeros((len(imu_data), 2))
counts = np.zeros(len(imu_data))

for i, start in enumerate(start_indices):
    full_pred_vel[start:start+window_size] += pred_vel_windows[i]
    counts[start:start+window_size] += 1

# Avoid divide by zero
counts[counts == 0] = 1
full_pred_vel /= counts[:, None]

# === Integrate velocity to get predicted position ===
dt = float(np.mean(np.diff(timestamps)))  # use mean dt as scalar
pred_pos = np.cumsum(full_pred_vel * dt, axis=0)



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
ate = compute_ATE(gt_pos, pred_pos)
t_rte = compute_T_RTE(gt_pos, pred_pos, dt=dt)
d_rte = compute_D_RTE(gt_pos, pred_pos)
pde = compute_PDE(gt_pos, pred_pos)

print(f"ATE:   {ate:.3f} m")
print(f"T-RTE: {t_rte:.3f} m")
print(f"D-RTE: {d_rte:.3f} m")
print(f"PDE:   {pde:.3f}")

# === Plot results ===
plt.figure(figsize=(8, 6))
plt.plot(gt_pos[:, 0], gt_pos[:, 1], label='Ground Truth', alpha=1)
plt.plot(pred_pos[:, 0], pred_pos[:, 1], label='CTIN Prediction', alpha=1)
plt.legend()
plt.title("CTIN Trajectory vs Ground Truth")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.axis('equal')
plt.grid(True)
plt.show()
