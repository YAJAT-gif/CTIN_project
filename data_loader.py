import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_and_process_data(imu_path, gt_path, window_size=200):
    # Load CSV files
    imu_df = pd.read_csv(imu_path)
    gt_df = pd.read_csv(gt_path)

    # Clean column names
    imu_df.rename(columns={"#timestamp [ns]": "timestamp"}, inplace=True)
    gt_df.rename(columns={"#timestamp": "timestamp"}, inplace=True)
    imu_df.columns = imu_df.columns.str.strip()
    gt_df.columns = gt_df.columns.str.strip()

    # Merge IMU and GT on timestamp
    merged_df = pd.merge(imu_df, gt_df, on="timestamp", how="inner")
    merged_df.columns = merged_df.columns.str.strip()

    # Extract quaternions and convert to rotation matrix
    quaternions = merged_df[["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"]].values
    rotations = R.from_quat(quaternions)

    # IMU in body frame
    gyro_body = merged_df[[
        "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]"
    ]].values
    accel_body = merged_df[[
        "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"
    ]].values

    # Rotate IMU to navigation frame
    gyro_nav = rotations.apply(gyro_body)
    accel_nav = rotations.apply(accel_body)

    # Concatenate rotated IMU data
    imu_nav_data = np.concatenate([gyro_nav, accel_nav], axis=1)

    # Ground truth velocity
    gt_velocity = merged_df[[
        "v_RS_R_x [m s^-1]", "v_RS_R_y [m s^-1]"
    ]].values

    # Sliding window
    X_windows = []
    Y_windows = []
    for i in range(len(imu_nav_data) - window_size + 1):
        X_windows.append(imu_nav_data[i:i + window_size])
        Y_windows.append(gt_velocity[i + window_size - 1])

    return np.array(X_windows), np.array(Y_windows)
