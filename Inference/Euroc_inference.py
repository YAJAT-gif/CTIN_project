# infer_euroc_ctin.py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


from ctin_project.model.ctin_model_gru import CTINModel

# =========================
# Config
# =========================
csv_path   = "../euroc_output/ctin_mh05.csv"
model_path = "../ctin_model_EUROC_GRU_3D.pth"
window_size = 200
stride      = 10
batch_size  = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalization safety
STD_FLOOR = 1e-3
IMU_CLIP  = 80.0    # pre-norm clamp (m/s^2 for acc, rad/s for gyro) – be generous
Z_CLIP    = 10.0    # post-norm clamp

# =========================
# Helpers
# =========================
def create_windows(data, window_size, stride):
    n = len(data)
    out = []
    for i in range(0, n - window_size + 1, stride):
        w = data[i:i+window_size]
        if w.shape[0] == window_size:
            out.append(w)
    return np.stack(out) if out else np.zeros((0, window_size, data.shape[1]), dtype=data.dtype)

def trapezoid_integrate(vel, t):
    """
    vel: [N, D] velocities at timestamps t (center cadence)
    t:   [N] seconds (monotonic)
    returns positions [N, D], starting at 0
    """
    N = vel.shape[0]
    pos = np.zeros_like(vel)
    if N <= 1:
        return pos
    dt = np.diff(t).astype(np.float64)
    dt = np.clip(dt, 1e-4, 1.0)  # guard
    incr = 0.5 * (vel[:-1] + vel[1:]) * dt[:, None]
    pos[1:] = np.cumsum(incr, axis=0)
    return pos

def plot_cdf(values, label, xlabel="ATE [m]", title="CDF of Absolute Trajectory Error"):
    v = np.sort(values)
    y = np.linspace(0, 1, len(v))
    plt.plot(v, y, label=label)
    plt.xlabel(xlabel); plt.ylabel("Cumulative probability")
    plt.grid(True); plt.title(title)

# =========================
# Load CSV
# =========================
df = pd.read_csv(csv_path).dropna()

required_cols = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','vx','vy','vz','timestamp']
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

has_gt_pos = set(['gt_x','gt_y','gt_z']).issubset(df.columns)

imu = df[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']].to_numpy(dtype=np.float32)
vel_gt = df[['vx','vy','vz']].to_numpy(dtype=np.float64)
t_all = df['timestamp'].to_numpy(dtype=np.float64)

# =========================
# Normalize IMU robustly (per file)
# =========================
imu = np.clip(imu, -IMU_CLIP, IMU_CLIP)
mean = imu.mean(axis=0, dtype=np.float64)
std  = imu.std(axis=0, dtype=np.float64)
std  = np.maximum(std, STD_FLOOR)
imu  = (imu - mean.astype(np.float32)) / std.astype(np.float32)
imu  = np.clip(imu, -Z_CLIP, Z_CLIP)

# =========================
# Build windows at stride, center indices, timestamps
# =========================
X_windows = create_windows(imu, window_size, stride)
if X_windows.shape[0] == 0:
    raise RuntimeError("No windows created; check window_size/stride vs file length.")

N_windows = X_windows.shape[0]
center_idx0 = window_size // 2
center_indices = np.arange(center_idx0, center_idx0 + stride * N_windows, stride)
assert center_indices[-1] < len(df), "Center indices exceed dataframe length."

t_center = t_all[center_indices]

# =========================
# Inference
# =========================
model = CTINModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

pred_chunks = []
with torch.no_grad():
    X_tensor = torch.tensor(X_windows, dtype=torch.float32, device=device)
    # Final guard
    X_tensor = torch.nan_to_num(X_tensor, nan=0.0, posinf=1e3, neginf=-1e3)
    for i in range(0, X_tensor.shape[0], batch_size):
        batch = X_tensor[i:i+batch_size]
        pred_vel, _ = model(batch)               # [B, T, 3]
        center = pred_vel[:, window_size//2, :]  # [B, 3]
        pred_chunks.append(center.detach().cpu())
center_pred = torch.cat(pred_chunks, dim=0)      # [N, 3]

# Optional: de-bias predicted velocity (helps drift)
center_pred = center_pred - center_pred.mean(dim=0, keepdim=True)

# =========================
# Integrate predictions (with timestamps)
# =========================
vel_pred_np = center_pred.numpy().astype(np.float64)
pos_pred = trapezoid_integrate(vel_pred_np, t_center)  # [N,3]

# =========================
# Align GT positions to centers (no integration)
# =========================
if has_gt_pos:
    pos_gt_all = df[['gt_x','gt_y','gt_z']].to_numpy(dtype=np.float64)
    pos_gt_aligned = pos_gt_all[center_indices] - pos_gt_all[center_indices[0]]
else:
    # Fallback: integrate GT velocity on same cadence (rarely needed with EuRoC)
    v_gt_c = vel_gt[center_indices]
    pos_gt_aligned = trapezoid_integrate(v_gt_c, t_center)

# =========================
# Metrics
# =========================
err_vec = pos_pred - pos_gt_aligned
err_norm = np.linalg.norm(err_vec, axis=1)
err_h = np.linalg.norm(err_vec[:, :2], axis=1)
err_v = np.abs(err_vec[:, 2])

ATE_mean = err_norm.mean()
ATE_med  = np.median(err_norm)
ATE_p95  = np.percentile(err_norm, 95)
RMSE     = np.sqrt(mean_squared_error(pos_gt_aligned, pos_pred))
Emax     = err_norm.max()

print("\nCTIN Evaluation (center-cadence):")
print(f"ATE mean    : {ATE_mean:.3f} m")
print(f"ATE median  : {ATE_med:.3f} m")
print(f"ATE p95     : {ATE_p95:.3f} m")
print(f"RMSE        : {RMSE:.3f} m")
print(f"Max error   : {Emax:.3f} m")

# =========================
# Poster-ready EuRoC plots
# =========================
# =========================
# EuRoC 2×2 Poster Panel (no 3D plot)
# =========================
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Colors (color-blind friendly)
COL_GT   = "#000000"  # black
COL_CTIN = "#E69F00"  # orange
COL_OV2  = "#56B4E9"  # light blue

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "savefig.dpi": 300
})

def _add_dir_arrow(ax, P, color):
    if len(P) < 2: return
    a = FancyArrowPatch((P[-2,0], P[-2,1]), (P[-1,0], P[-1,1]),
                        arrowstyle="-|>", mutation_scale=10, lw=0, color=color, zorder=6)
    ax.add_patch(a)

# Normalize positions to common origin for overlay
GT0 = pos_gt_aligned - pos_gt_aligned[0]
PR0 = pos_pred       - pos_gt_aligned[0]
t_rel = t_center - t_center[0]

fig = plt.figure(figsize=(7.2, 5.4))
fig.suptitle("EuRoC Dataset (UAV)", y=0.98, fontsize=14, fontweight="bold")

# ---- (1) CDF of ATE ----
ax1 = fig.add_subplot(2,2,1)
s = np.sort(err_norm); y = np.linspace(0,1,len(s))
ax1.plot(s, y, color=COL_CTIN, lw=2, label="CTIN")
p50 = float(np.percentile(err_norm, 50))
ax1.axvline(p50, color=COL_CTIN, ls=":", lw=1)
ax1.text(p50, 0.05, f"median={p50:.2f} m", rotation=90, va="bottom", ha="right", fontsize=8)
ax1.axvline(ATE_p95, color="k", ls="--", lw=1)
ax1.text(ATE_p95, 0.1, f"p95={ATE_p95:.2f} m", rotation=90, va="bottom", ha="right", fontsize=8)
ax1.set_xlabel("ATE [m]"); ax1.set_ylabel("Cumulative probability")
ax1.set_title("CDF of ATE"); ax1.grid(True, alpha=0.3); ax1.legend(frameon=False, loc="lower left")

# ---- (2) Error vs Time ----
ax2 = fig.add_subplot(2,2,2)
ax2.plot(t_rel, err_norm, color=COL_CTIN, lw=2, label="Position error")
ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Error [m]")
ax2.set_title("Error vs Time"); ax2.grid(True, alpha=0.3); ax2.legend(frameon=False, loc="upper right")

# ---- (3) Top-down Trajectory (XY) ----
ax3 = fig.add_subplot(2,2,3)
ax3.plot(GT0[:,0], GT0[:,1], color=COL_GT,   lw=2.2, label="GT")
ax3.plot(PR0[:,0], PR0[:,1], color=COL_CTIN, lw=2.0, ls="--", label="CTIN")
ax3.scatter([GT0[0,0]],[GT0[0,1]], c=COL_GT,   s=24, marker="o")
ax3.scatter([PR0[0,0]],[PR0[0,1]], c=COL_CTIN, s=22, marker="o")
ax3.scatter([GT0[-1,0]],[GT0[-1,1]], c=COL_GT,   s=30, marker="X")
ax3.scatter([PR0[-1,0]],[PR0[-1,1]], c=COL_CTIN, s=28, marker="X")
_add_dir_arrow(ax3, PR0, COL_CTIN)
ax3.set_aspect("equal", adjustable="box"); ax3.margins(0.03)
ax3.set_xlabel("X [m]"); ax3.set_ylabel("Y [m]")
ax3.set_title("Top-down Trajectory (XY)")
ax3.grid(True, alpha=0.25); ax3.legend(frameon=False, loc="lower left")

# ---- (4) RMSE Bar Comparison ----
ax4 = fig.add_subplot(2,2,4)
ctin_rmse = float(RMSE)  # your measured CTIN RMSE
ov2slam_rmse = 0.192     # OV²SLAM-Fast 200 Hz on MH_05 from paper
labels = ["CTIN", "OV²SLAM-Fast\n(200 Hz)"]
vals   = [ctin_rmse, ov2slam_rmse]
colors = [COL_CTIN, COL_OV2]
bars = ax4.bar(np.arange(2), vals, color=colors, alpha=0.95)
for b, v in zip(bars, vals):
    ax4.text(b.get_x()+b.get_width()/2, v + 0.02*max(vals), f"{v:.3f}",
             ha="center", va="bottom", fontsize=8)
ax4.set_xticks(np.arange(2)); ax4.set_xticklabels(labels, rotation=0)
ax4.set_ylabel("ATE RMSE [m]")
ax4.set_title("RMSE Comparison (MH_05)")
ax4.grid(True, axis="y", alpha=0.3)

fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("euroc_uav_2x2_with_bar.png")
fig.savefig("euroc_uav_2x2_with_bar.svg")
plt.show()
