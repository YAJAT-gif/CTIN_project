import numpy as np
import torch

# === CONFIGURATION ===
fs = 100  # Sampling frequency (Hz)
dt = 1.0 / fs
window_size = 200

# Choose steps for each segment
steps = {
    "A": 100,   # Stationary
    "B": 200,   # Constant forward acceleration
    "C": 150,   # Constant velocity
    "D": 1000,   # Turning (yaw)
    "E": 150,   # Oscillatory acceleration
    "F": 100    # Stationary again
}

# === HELPERS ===
def integrate_accel(accel, dt, v0=np.array([0.0, 0.0])):
    velocity = np.cumsum(accel, axis=0) * dt
    return velocity + v0

# === SEGMENTS ===
segments = []

# A: Zero velocity (static)
imu_A = np.zeros((steps["A"], 6))
v_A = np.zeros((steps["A"], 2))
segments.append((imu_A, v_A))

# B: Constant forward acceleration
accel_val = 1.0  # m/s²
imu_B = np.zeros((steps["B"], 6))
imu_B[:, 3] = accel_val  # α_x
v_B = integrate_accel(imu_B[:, 3:5], dt)
segments.append((imu_B, v_B))

# C: Constant velocity
imu_C = np.zeros((steps["C"], 6))
v_C = np.tile(v_B[-1], (steps["C"], 1))  # carry last vel
segments.append((imu_C, v_C))

# D: Turning motion
imu_D = np.zeros((steps["D"], 6))
yaw_rate = np.pi / 100  # rad/s
imu_D[:, 2] = yaw_rate  # ω_z
v_D = np.zeros((steps["D"], 2))
speed = np.linalg.norm(v_C[0])
theta = 0
for i in range(steps["D"]):
    theta += yaw_rate * dt
    v_D[i] = [speed * np.cos(theta), speed * np.sin(theta)]
segments.append((imu_D, v_D))

# E: Oscillatory acceleration
imu_E = np.zeros((steps["E"], 6))
freq = 1.0  # Hz
A = 1.0     # m/s²
t_E = np.arange(steps["E"]) * dt
imu_E[:, 3] = A * np.sin(2 * np.pi * freq * t_E)
v_E = integrate_accel(imu_E[:, 3:5], dt, v0=v_D[-1])
segments.append((imu_E, v_E))

# F: Zero velocity again
imu_F = np.zeros((steps["F"], 6))
v_F = np.zeros((steps["F"], 2))
segments.append((imu_F, v_F))

# === CONCATENATE SEGMENTS ===
imu_all = np.concatenate([s[0] for s in segments], axis=0)
v_all = np.concatenate([s[1] for s in segments], axis=0)

torch.save(torch.tensor(v_all, dtype=torch.float32), "Y_full.pt")  # [T, 2]


# === ROLLING WINDOWS ===
X = []
Y = []
for i in range(len(imu_all) - window_size + 1):
    X.append(imu_all[i:i + window_size])
    Y.append(v_all[i:i + window_size])

X = torch.tensor(np.stack(X), dtype=torch.float32)
Y = torch.tensor(np.stack(Y), dtype=torch.float32)

#=== SAVE ===
torch.save(X, "X_synthetic_loop.pt")
torch.save(Y, "Y_synthetic_loop.pt")

print(f"Saved synthetic dataset:")
print(f"X shape: {X.shape} (IMU window)")
print(f"Y shape: {Y.shape} (Velocity window)")

import torch
import matplotlib.pyplot as plt
import numpy as np

# Load velocity data
Y = torch.load("Y_synthetic_loop.pt")  # shape: [num_windows, 200, 2]
vel = Y[0]  # first window for example

# Optional: use entire sequence if you want full path
# This assumes continuous overlapping windows
# Stitch velocity sequence from the full data:
# (i.e., the first velocity of each window gives a clean sequence)
vel_seq = Y[:, 0, :].numpy()

# Integrate velocity to get position
dt = 1.0 / 100  # 100 Hz

# Plot trajectory
plt.figure(figsize=(6, 6))
plt.plot(pos[:, 0], pos[:, 1], marker='o', markersize=2, linewidth=1)
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Synthetic Trajectory (Integrated from Velocity)")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()
