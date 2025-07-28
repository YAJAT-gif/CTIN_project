import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from ctin_project.model.ctin_model import CTINModel

# === Config ===
csv_path = "ctin_dataset_137102747096458.csv"
model_path = "../ctin_model_tlio.pth"
window_size = 200
stride = 10
batch_size = 8
dt = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load and normalize IMU ===
df = pd.read_csv(csv_path).dropna()
imu = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
vel_gt = df[['vel_x', 'vel_y']].values

imu_mean = imu.mean(axis=0)
imu_std = imu.std(axis=0) + 1e-6
imu = (imu - imu_mean) / imu_std

# === Create sliding windows ===
def create_windows(data, window_size, stride):
    return np.stack([
        data[i:i + window_size]
        for i in range(0, len(data) - window_size + 1, stride)
        if data[i:i + window_size].shape[0] == window_size
    ])

X_windows = create_windows(imu, window_size, stride)
X_tensor = torch.tensor(X_windows, dtype=torch.float32).to(device)

# === Load CTIN model ===
model = CTINModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Inference: center velocity only, batched ===
center_preds = []
with torch.no_grad():
    for i in range(0, X_tensor.shape[0], batch_size):
        batch = X_tensor[i:i + batch_size]
        pred_vel, _ = model(batch)
        center = pred_vel[:, window_size // 2, :]  # center frame
        center_preds.append(center.cpu())

center_pred = torch.cat(center_preds, dim=0)  # shape [N, 2]

# === Integrate predicted velocity ===
def integrate_velocity(vel, dt):
    pos = torch.zeros_like(vel)
    pos[0] = vel[0] * dt
    for t in range(1, vel.shape[0]):
        pos[t] = pos[t - 1] + vel[t - 1] * dt
    return pos

pos_pred = integrate_velocity(center_pred, dt=dt).numpy()

# === Align ground truth velocity for fair comparison ===
start = window_size // 2
center_gt = vel_gt[start:start + stride * len(center_pred):stride]
pos_gt_aligned = integrate_velocity(torch.tensor(center_gt), dt=dt).numpy()

# === Compute error metrics ===
errors = np.linalg.norm(pos_pred - pos_gt_aligned, axis=1)
ate = np.mean(errors)
rmse = np.sqrt(mean_squared_error(pos_gt_aligned, pos_pred))

# === Print evaluation ===
print(f"\nCTIN Evaluation Results:")
print(f"ATE (Mean Position Error): {ate:.4f} meters")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f} meters")
print(f"Max Position Error: {np.max(errors):.4f} meters")

# === Velocity stats ===
print("\nPredicted velocity stats:")
print("Mean:", center_pred.mean(dim=0))
print("Std:", center_pred.std(dim=0))
print("Max:", center_pred.max(dim=0).values)
print("Min:", center_pred.min(dim=0).values)

# === Plotting ===
plt.figure(figsize=(8, 6))
plt.plot(pos_gt_aligned[:, 0], pos_gt_aligned[:, 1], label="GT Trajectory", linewidth=2)
plt.plot(pos_pred[:, 0], pos_pred[:, 1], label="CTIN Predicted", linestyle='--')
plt.title("CTIN vs Ground Truth Trajectory")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
