# infer_tlio_ctin_save_npz.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch

# --- your project imports (adjust if your paths differ) ---
from ctin_project.model.ctin_model import CTINModel  # must return (vel, logstd) with shape [B,T,D]

# =========================
# Config (edit these)
# =========================
csv_path    = Path("../ctin_csv_output/validation/ctin_dataset_137102747096458.csv")  # one TLIO_golden CSV
model_path  = Path("../ctin_model_tlio_GRU_NLL.pth")              # your trained checkpoint
save_npz    = Path("../Inference/TLIO/137102747096458/ctin_results.npz")

# Windowing / batching
window_size = 200
stride      = 20
batch_size  = 256

# IMU normalization (match training)
STD_FLOOR = 1e-3
IMU_CLIP  = 80.0
Z_CLIP    = 10.0

# Expected columns in TLIO_golden CSV (adjust if needed)
IMU_COLS   = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
VEL_COLS   = ["vel_x","vel_y"]              # 2‑D velocity (common for RoNIN)
POS_COLS   = ["gt_x","gt_y"]          # optional (if present)
TIME_COL   = "timestamp"              # seconds; or any monotonic timebase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Helpers
# =========================
def create_windows(data, window, stride):
    n = len(data)
    out = []
    for i in range(0, n - window + 1, stride):
        w = data[i:i+window]
        if w.shape[0] == window:
            out.append(w)
    if not out:
        return np.zeros((0, window, data.shape[1]), dtype=data.dtype)
    return np.stack(out)

def trapezoid_integrate(vel, t):
    """
    vel [N,D], t [N] -> pos [N,D], start at 0. Integrate on the provided cadence.
    """
    N = vel.shape[0]
    pos = np.zeros_like(vel)
    if N <= 1: return pos
    dt = np.diff(t).astype(np.float64)
    dt = np.clip(dt, 1e-4, 1.0)
    incr = 0.5 * (vel[:-1] + vel[1:]) * dt[:, None]
    pos[1:] = np.cumsum(incr, axis=0)
    return pos

# =========================
# Load CSV
# =========================
df = pd.read_csv(csv_path).dropna()
for c in IMU_COLS + VEL_COLS + [TIME_COL]:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

has_gt_pos = all(c in df.columns for c in POS_COLS)

imu = df[IMU_COLS].to_numpy(dtype=np.float32)   # [N,6]
vel_gt_all = df[VEL_COLS].to_numpy(dtype=np.float64)  # [N,2]
t_all = df[TIME_COL].to_numpy(dtype=np.float64)

# =========================
# Normalize IMU (per-file, match training)
# =========================
imu = np.clip(imu, -IMU_CLIP, IMU_CLIP)
m = imu.mean(axis=0, dtype=np.float64)
s = np.maximum(imu.std(axis=0, dtype=np.float64), STD_FLOOR)
imu = (imu - m.astype(np.float32)) / s.astype(np.float32)
imu = np.clip(imu, -Z_CLIP, Z_CLIP)

# =========================
# Windows & center cadence
# =========================
X_windows = create_windows(imu, window_size, stride)                # [W, T, 6]
if X_windows.shape[0] == 0:
    raise RuntimeError("No windows created; adjust window_size/stride vs file length.")

Nw = X_windows.shape[0]
mid = window_size // 2
center_idx = np.arange(mid, mid + stride*Nw, stride)
if center_idx[-1] >= len(df):
    # Robust guard if the last jump overflows by 1–2 samples
    center_idx = center_idx[center_idx < len(df)]
    X_windows = X_windows[:len(center_idx)]
Nw = X_windows.shape[0]

t_center = t_all[center_idx]
vel_gt   = vel_gt_all[center_idx]                                 # [Nw,2]

# If GT pos exists, pick center-cadence GT positions; else integrate GT velocity
if has_gt_pos:
    pos_gt_all = df[POS_COLS].to_numpy(dtype=np.float64)
    pos_gt = pos_gt_all[center_idx] - pos_gt_all[center_idx[0]]   # origin at 0
else:
    pos_gt = trapezoid_integrate(vel_gt, t_center)

# =========================
# Inference
# =========================
model = CTINModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

vel_list, logstd_list = [], []
with torch.no_grad():
    X = torch.tensor(X_windows, dtype=torch.float32, device=device)   # [W,T,6]
    X = torch.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
    for i in range(0, X.shape[0], batch_size):
        pv, pl = model(X[i:i+batch_size])   # pv/pl: [B,T,2]
        vel_list.append(pv[:, mid, :].detach().cpu())     # center‑cadence velocity
        logstd_list.append(pl[:, mid, :].detach().cpu())  # center‑cadence logσ

vel_pred_center = torch.cat(vel_list, dim=0).numpy().astype(np.float64)   # [Nw,2]
logstd_vel      = torch.cat(logstd_list, dim=0).numpy().astype(np.float64) # [Nw,2]
sigma_v         = np.exp(logstd_vel).astype(np.float64)                    # [Nw,2]

# (Optional) small de-bias of mean velocity over the file
vel_pred_center = vel_pred_center - vel_pred_center.mean(axis=0, keepdims=True)

# =========================
# Positions for quick metrics (integrate center‑cadence vel)
# =========================
pos_pred = trapezoid_integrate(vel_pred_center, t_center)                 # [Nw,2]
pos_gt0  = pos_gt - pos_gt[0]                                            # [Nw,2]
pos_pr0  = pos_pred - pos_pred[0]

# =========================
# Save NPZ for downstream plotting
# =========================
os.makedirs(save_npz.parent, exist_ok=True)
np.savez_compressed(
    save_npz,
    t=t_center.astype(np.float64),          # [Nw]
    pos_gt=pos_gt0.astype(np.float64),      # [Nw,2]
    pos_pred=pos_pr0.astype(np.float64),    # [Nw,2]
    vel_gt=vel_gt.astype(np.float64),       # [Nw,2]
    vel_pred=vel_pred_center.astype(np.float64),  # [Nw,2]
    logstd_vel=logstd_vel.astype(np.float64),     # [Nw,2]
    sigma_v=sigma_v.astype(np.float64)            # [Nw,2]
)
print(f"[OK] Saved: {save_npz}  (keys: t, pos_gt, pos_pred, vel_gt, vel_pred, logstd_vel, sigma_v)")

# =========================
# Quick console metrics
# =========================
e_vec  = pos_pr0 - pos_gt0
e_norm = np.linalg.norm(e_vec, axis=1)
mean   = float(e_norm.mean())
p95    = float(np.percentile(e_norm, 95))
rmse   = float(np.sqrt(np.mean((pos_pr0 - pos_gt0) ** 2)))
emax   = float(e_norm.max())

print("CTIN (center‑cadence) ATE:")
print(f"  mean {mean:.3f} m | p95 {p95:.3f} m | RMSE {rmse:.3f} m | max {emax:.3f} m")

# Optional: quick velocity z‑coverage print
z = (vel_gt - vel_pred_center) / (sigma_v + 1e-8)
c1 = float((np.abs(z) <= 1.0).mean()) * 100.0
c2 = float((np.abs(z) <= 2.0).mean()) * 100.0
c3 = float((np.abs(z) <= 3.0).mean()) * 100.0
print(f"Velocity coverage: ≤1σ {c1:.1f}%  ≤2σ {c2:.1f}%  ≤3σ {c3:.1f}%")
