import numpy as np
import torch
import matplotlib.pyplot as plt

# === Parameters ===
fs = 100
dt = 1 / fs
window_size = 200

def integrate_accel(accel, dt, v0=np.array([0., 0.])):
    return np.cumsum(accel, axis=0) * dt + v0

# === Segment builder ===
segments = []

# A: Stationary
steps_A = 100
imu_A = np.zeros((steps_A, 6))
v_A = np.zeros((steps_A, 2))
segments.append((imu_A, v_A))

# B: Accelerate forward (X-axis)
steps_B = 200
imu_B = np.zeros((steps_B, 6)); imu_B[:, 3] = 0.4
v_B = integrate_accel(imu_B[:, 3:5], dt)
segments.append((imu_B, v_B))

# C: Left 90° turn
steps_C = 250
imu_C = np.zeros((steps_C, 6))
imu_C[:, 2] = np.pi / 125
speed = np.linalg.norm(v_B[-1])
v_C = np.zeros((steps_C, 2))
theta = 0
for i in range(steps_C):
    theta += imu_C[i, 2] * dt
    v_C[i] = [speed * np.cos(theta), speed * np.sin(theta)]
segments.append((imu_C, v_C))

# D: Straight top side
steps_D = 200
imu_D = np.zeros((steps_D, 6))
v_D = np.tile(v_C[-1], (steps_D, 1))
segments.append((imu_D, v_D))

# E: Left 90° turn (heading down)
steps_E = 250
imu_E = np.zeros((steps_E, 6))
imu_E[:, 2] = np.pi / 125
v_E = np.zeros((steps_E, 2))
theta = np.pi / 2
for i in range(steps_E):
    theta += imu_E[i, 2] * dt
    v_E[i] = [speed * np.cos(theta), speed * np.sin(theta)]
segments.append((imu_E, v_E))

# F: Return straight
steps_F = 200
imu_F = np.zeros((steps_F, 6))
v_F = np.tile(v_E[-1], (steps_F, 1))
segments.append((imu_F, v_F))

# G: Final left 90° turn
steps_G = 250
imu_G = np.zeros((steps_G, 6))
imu_G[:, 2] = np.pi / 125
v_G = np.zeros((steps_G, 2))
theta = np.pi
for i in range(steps_G):
    theta += imu_G[i, 2] * dt
    v_G[i] = [speed * np.cos(theta), speed * np.sin(theta)]
segments.append((imu_G, v_G))

# H: Final straight back to origin
steps_H = 200
imu_H = np.zeros((steps_H, 6))
v_H = np.tile(v_G[-1], (steps_H, 1))
segments.append((imu_H, v_H))

# I: Stationary end
steps_I = 100
imu_I = np.zeros((steps_I, 6))
v_I = np.zeros((steps_I, 2))
segments.append((imu_I, v_I))

# === Merge all ===
imu_all = np.concatenate([s[0] for s in segments])
v_all = np.concatenate([s[1] for s in segments])
pos_all = np.cumsum(v_all * dt, axis=0)

from scipy.signal import butter, filtfilt

def low_freq_noise(length, std, cutoff=0.05, fs=100):
    noise = np.random.normal(0, std, size=(length,))
    b, a = butter(2, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, noise)

# === Add noise ===
imu_noisy = imu_all.copy()

# Add white noise
imu_noisy[:, 0:3] += np.random.normal(0, 0.03, imu_noisy[:, 0:3].shape)  # gyro
imu_noisy[:, 3:6] += np.random.normal(0, 0.4, imu_noisy[:, 3:6].shape)   # accel

# Add low-frequency drift
for i in range(6):
    std = 0.2 if i < 3 else 0.1  # gyro vs accel
    imu_noisy[:, i] += low_freq_noise(len(imu_noisy), std=std, cutoff=0.05)


# === Create windowed data ===
X, Y = [], []
for i in range(len(imu_all) - window_size + 1):
    X.append(imu_noisy[i:i + window_size])
    Y.append(v_all[i:i+window_size])
X_tensor = torch.tensor(np.stack(X), dtype=torch.float32)
Y_tensor = torch.tensor(np.stack(Y), dtype=torch.float32)
Y_full = torch.tensor(v_all, dtype=torch.float32)

# === Save ===
torch.save(X_tensor, "X_circuit.pt")
torch.save(Y_tensor, "Y_circuit.pt")
torch.save(Y_full, "Y_full_circuit.pt")
print("Saved: X_circuit.pt, Y_circuit.pt, Y_full_circuit.pt")

# === Plot ===
plt.figure(figsize=(8, 8))
plt.plot(pos_all[:, 0], pos_all[:, 1], label="Closed Circuit Trajectory", linewidth=2)
plt.axis("equal")
plt.grid(True)
plt.title("Synthetic Closed-Loop Circuit Trajectory")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(imu_all[:, 3], label="Clean ax")
plt.grid(); plt.show()
plt.plot(imu_noisy[:, 3], label="Noisy ax")
plt.legend(); plt.title("Accel X Before vs After Noise")
plt.grid(); plt.show()
