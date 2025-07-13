import numpy as np
import torch
from scipy.signal import butter, filtfilt

# Parameters
fs = 100
dt = 1 / fs
window_size = 200

def integrate_accel(accel, dt, v0=np.array([0., 0.])):
    return np.cumsum(accel, axis=0) * dt + v0

def low_freq_noise(length, std, cutoff=0.1, fs=100):
    noise = np.random.normal(0, std, size=(length,))
    b, a = butter(2, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, noise)

# Segment setup
steps = {"A": 100, "B": 200, "C": 550, "D": 500, "E": 550, "F": 200}
segments = []

# Segment A: Stationary
segments.append((np.zeros((steps["A"], 6)), np.zeros((steps["A"], 2))))

# Segment B: Acceleration
imu_B = np.zeros((steps["B"], 6)); imu_B[:, 3] = 1.0
v_B = integrate_accel(imu_B[:, 3:5], dt)
segments.append((imu_B, v_B))

# Segment C: Constant velocity
imu_C = np.zeros((steps["C"], 6))
v_C = np.tile(v_B[-1], (steps["C"], 1))
segments.append((imu_C, v_C))

# Segment D: Turning
imu_D = np.zeros((steps["D"], 6)); imu_D[:, 2] = np.pi / 50
v_D = np.zeros((steps["D"], 2)); theta = 0
speed = np.linalg.norm(v_C[0])
for i in range(steps["D"]):
    theta += imu_D[i, 2] * dt
    v_D[i] = [speed * np.cos(theta), speed * np.sin(theta)]
segments.append((imu_D, v_D))

# Segment E: Oscillatory
imu_E = np.zeros((steps["E"], 6))
imu_E[:, 3] = np.sin(2 * np.pi * 1.0 * np.arange(steps["E"]) * dt)
v_E = integrate_accel(imu_E[:, 3:5], dt, v_D[-1])
segments.append((imu_E, v_E))

# Segment F: Stationary
segments.append((np.zeros((steps["F"], 6)), np.zeros((steps["F"], 2))))

# Merge segments
imu_all = np.concatenate([s[0] for s in segments])
v_all = np.concatenate([s[1] for s in segments])

# Add noise
imu_noisy = imu_all.copy()
imu_noisy[:, 0:3] += np.random.normal(0, 0.03, imu_noisy[:, 0:3].shape)
imu_noisy[:, 3:6] += np.random.normal(0, 0.4, imu_noisy[:, 3:6].shape)
for i in range(6):
    imu_noisy[:, i] += low_freq_noise(len(imu_noisy), std=0.2 if i < 3 else 0.1)

# Windowed data
X_noisy, Y_windowed = [], []
for i in range(len(imu_noisy) - window_size + 1):
    X_noisy.append(imu_noisy[i:i+window_size])
    Y_windowed.append(v_all[i:i+window_size])

# Save
torch.save(torch.tensor(np.stack(X_noisy), dtype=torch.float32), "X_synthetic_noisy.pt")
torch.save(torch.tensor(np.stack(Y_windowed), dtype=torch.float32), "Y_synthetic_noisy.pt")
torch.save(torch.tensor(v_all, dtype=torch.float32), "Y_full_noisy.pt")
print("Noisy dataset saved.")

import torch
import numpy as np
import matplotlib.pyplot as plt

# === Config ===
dt = 1 / 100  # 100 Hz sampling rate

# === Load full velocity (ground truth, noisy motion path) ===
vel_true = torch.load("Y_full_noisy.pt").numpy()

# === Integrate velocity to get position ===
pos_true = np.cumsum(vel_true * dt, axis=0)

# === Plot the trajectory ===
plt.figure(figsize=(8, 8))
plt.plot(pos_true[:, 0], pos_true[:, 1], label="Noisy Ground Truth Trajectory", linewidth=2)
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Trajectory from Noisy Synthetic Data")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(imu_all[:, 3], label="Clean ax")
plt.grid(); plt.show()
plt.plot(imu_noisy[:, 3], label="Noisy ax")
plt.legend(); plt.title("Accel X Before vs After Noise")
plt.grid(); plt.show()
