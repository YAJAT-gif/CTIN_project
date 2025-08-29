import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# === Config ===
euroc_root = r"D:/MH_01_easy/mav0"  # <-- CHANGE THIS
imu_path = os.path.join(euroc_root, "imu0", "data.csv")
gt_path = os.path.join(euroc_root, "state_groundtruth_estimate0", "data.csv")
out_csv_path = "../euroc_output/ctin_mh01.csv"

# === Load data ===
imu_df = pd.read_csv(imu_path, comment='#', header=None)
gt_df = pd.read_csv(gt_path, comment='#', header=None)

# === Assign columns ===
imu_df.columns = ["timestamp", "gx", "gy", "gz", "ax", "ay", "az"]
gt_df.columns = [
    "timestamp",
    "px", "py", "pz",
    "qw", "qx", "qy", "qz",
    "vx", "vy", "vz",
    "bw_x", "bw_y", "bw_z",
    "ba_x", "ba_y", "ba_z"
]

# === Convert timestamps from ns to seconds ===
imu_df['timestamp'] = imu_df['timestamp'] / 1e9
gt_df['timestamp'] = gt_df['timestamp'] / 1e9

# === Interpolate GT to IMU timestamps ===
interp_gt = {}
for col in ["vx", "vy","vz", "px", "py","pz"]:
    interp_gt[col] = np.interp(imu_df['timestamp'], gt_df['timestamp'], gt_df[col])

# === Create output dataframe ===
final_df = pd.DataFrame({
    "timestamp": imu_df['timestamp'],
    "acc_x": imu_df["ax"],
    "acc_y": imu_df["ay"],
    "acc_z": imu_df["az"],
    "gyro_x": imu_df["gx"],
    "gyro_y": imu_df["gy"],
    "gyro_z": imu_df["gz"],
    "vx": interp_gt["vx"],
    "vy": interp_gt["vy"],
    "vz": interp_gt["vz"],
    "gt_x": interp_gt["px"],
    "gt_y": interp_gt["py"],
    "gt_z": interp_gt["pz"]
})

# === Save CSV ===
final_df.to_csv(out_csv_path, index=False)
print(f"Saved CSV to {out_csv_path}")

# === Plot ground truth trajectory ===
# plt.figure(figsize=(8, 6))
# plt.plot(interp_gt["px"], interp_gt["py"],interp_gt["pz"], label="GT Trajectory", linewidth=2)
# plt.xlabel("X [m]")
# plt.ylabel("Y [m]")
# plt.label("Z [m]")
# plt.title("Ground Truth Trajectory from EuRoC")
# plt.axis("equal")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(interp_gt["px"], interp_gt["py"],interp_gt["pz"], label="GT Trajectory", linewidth=2)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.legend()
plt.tight_layout()
plt.show()