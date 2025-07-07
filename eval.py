import torch
import numpy as np
import matplotlib.pyplot as plt
from model.ctin_model import CTINModel

# Load model
model = CTINModel()
model.load_state_dict(torch.load("ctin_model.pth", map_location=torch.device("cpu")))
model.eval()

# Load IMU and ground truth data
imu_data = np.loadtxt("imu_data.csv", delimiter=",", skiprows=1, usecols=(1, 2, 3, 4, 5, 6))
gt_data = np.loadtxt("gt_data.csv", delimiter=",", skiprows=1, usecols=(7, 8))

# Prepare sliding windows
window_size = 200
imu_windows = []
gt_velocities = []

num_windows = len(imu_data) - window_size + 1

for i in range(num_windows):
    imu_slice = imu_data[i:i + window_size]
    vel_slice = gt_data[i:i + window_size]

    if imu_slice.shape[0] == window_size and vel_slice.shape[0] == window_size:
        imu_windows.append(imu_slice)
        gt_velocities.append(vel_slice)

imu_windows = np.array(imu_windows)             # Shape: [N, 200, 6]
gt_velocity = np.array(gt_velocities)           # Shape: [N, 200, 2]


X_tensor = torch.tensor(np.array(imu_windows), dtype=torch.float32)


# Visualize 5 samples
dt = 0.01
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for i in range(5):
    print(f"Sample {i} mean:", imu_windows[i].mean(), "std:", imu_windows[i].std())
    with torch.no_grad():
        pred_vel, _ = model(X_tensor[i:i+1])

    pos_pred = np.zeros((window_size, 2))
    pos_gt = np.zeros((window_size, 2))
    for t in range(1, window_size):
        pos_pred[t] = pos_pred[t - 1] + pred_vel[0, t - 1].numpy() * dt
        pos_gt[t] = pos_gt[t - 1] + gt_velocity[i, t - 1] * dt

    axs[i].plot(pos_gt[:, 0], pos_gt[:, 1], label="GT", color="blue")
    axs[i].plot(pos_pred[:, 0], pos_pred[:, 1], label="Pred", color="orange")
    axs[i].set_title(f"Sample {i}")
    axs[i].set_xlabel("X (m)")
    if i == 0:
        axs[i].set_ylabel("Y (m)")
    axs[i].axis("equal")
    axs[i].grid(True)

fig.suptitle("Predicted vs Ground Truth Trajectories (First 5 Samples)", fontsize=14)
fig.legend(["Ground Truth", "Predicted"], loc="upper right")
plt.tight_layout()
plt.show()
