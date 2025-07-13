import h5py
import pandas as pd
import matplotlib.pyplot as plt
import os

def convert_ronin_synced_to_csv(h5_path, output_csv_path):
    with h5py.File(h5_path, 'r') as f:
        timestamps = f['synced/time'][:]          # shape: (N,)
        acc = f['synced/acce'][:]                 # shape: (N, 3)
        gyro = f['synced/gyro'][:]                # shape: (N, 3)
        pos = f['pose/tango_pos'][:]              # shape: (N, 3)

        # Build DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'acc_x': acc[:, 0],
            'acc_y': acc[:, 1],
            'acc_z': acc[:, 2],
            'gyro_x': gyro[:, 0],
            'gyro_y': gyro[:, 1],
            'gyro_z': gyro[:, 2],
            'gt_x': pos[:, 0],
            'gt_y': pos[:, 1],
        })

        df.to_csv(output_csv_path, index=False)
        print(f"✅ Saved CSV to: {output_csv_path}")

        # Plot GT trajectory
        plt.figure(figsize=(6, 6))
        plt.plot(df['gt_x'], df['gt_y'], label='Ground Truth Trajectory')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.grid(True)
        plt.title("RoNIN Ground Truth Trajectory")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # CHANGE THESE BEFORE RUNNING
    h5_file_path = "data.hdf5"
    output_csv_path = "output.csv"

    convert_ronin_synced_to_csv(h5_file_path, output_csv_path)
