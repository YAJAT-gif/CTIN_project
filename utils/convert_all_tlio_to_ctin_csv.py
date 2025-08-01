import os
import numpy as np
import pandas as pd

# === CONFIG ===
data_root = "local_data_1/tlio_golden"
train_list_path = "local_data_1/tlio_golden/train_list.txt"
output_dir = "../ctin_csv_output"

os.makedirs(output_dir, exist_ok=True)

# === Load train list entries ===
with open(train_list_path, "r") as f:
    sequence_list = [line.strip() for line in f if line.strip()]

# === Convert each sequence ===
for seq_rel_path in sequence_list:
    seq_id = os.path.basename(seq_rel_path)
    npy_path = os.path.join(data_root, seq_rel_path, "imu0_resampled.npy")

    if not os.path.isfile(npy_path):
        print(f"⚠️  Skipping {seq_id}: imu0_resampled.npy not found.")
        continue

    try:
        data = np.load(npy_path)

        timestamps = data[:, 0] * 1e-6  # µs → s
        gyro = data[:, 1:4]
        acc = data[:, 4:7]
        vel = data[:, 14:16]  # vx, vy only

        ctin_data = np.hstack([timestamps.reshape(-1, 1), acc, gyro, vel])
        columns = [
            "timestamp",
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "vel_x", "vel_y"
        ]

        df = pd.DataFrame(ctin_data, columns=columns)

        output_csv = os.path.join(output_dir, f"ctin_dataset_{seq_id}.csv")
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved: {output_csv}")

    except Exception as e:
        print(f"❌ Error processing {seq_id}: {e}")
