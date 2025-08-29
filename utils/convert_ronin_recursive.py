import os
import h5py
import json
import numpy as np
import pandas as pd

# === CONFIG ===
root_dir = "D:/unseen_subjects_test_set"            # RoNIN dataset root folder
save_dir = "D:/ronin_csvs_test"                 # Output CSVs here
os.makedirs(save_dir, exist_ok=True)

# === CONVERT FUNCTION ===
for seq_name in os.listdir(root_dir):
    seq_path = os.path.join(root_dir, seq_name)
    hdf5_path = os.path.join(seq_path, "data.hdf5")

    if not os.path.exists(hdf5_path):
        print(f"❌ Skipping {seq_name}: no data.hdf5")
        continue

    try:
        with h5py.File(hdf5_path, "r") as f:
            print(f"\n📂 Processing: {seq_name}")
            # Check keys
            if not all(k in f for k in ["synced", "pose"]):
                print(f"⚠️ Skipping {seq_name}: missing 'synced' or 'pose'")
                continue

            acc = f["synced/acce"][:]
            gyro = f["synced/gyro"][:]
            time = f["synced/time"][:]
            pos = f["pose/tango_pos"][:]

            # Validate length match
            min_len = min(len(time), len(acc), len(gyro), len(pos))
            acc = acc[:min_len]
            gyro = gyro[:min_len]
            time = time[:min_len]
            pos = pos[:min_len]

            # x, y only
            pos_2d = pos[:, :2]

            data = np.hstack([
                time[:, None],
                acc,
                gyro,
                pos_2d
            ])

            df = pd.DataFrame(data, columns=[
                "timestamp",
                "acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z",
                "gt_x", "gt_y"
            ])

            save_path = os.path.join(save_dir, f"{seq_name}.csv")
            df.to_csv(save_path, index=False)
            print(f"✅ Saved: {save_path}")

    except Exception as e:
        print(f"❌ Failed on {seq_name}: {e}")
