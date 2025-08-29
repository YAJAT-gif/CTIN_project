# infer_euroc_ctin_with_cov_aligned.py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib.patches import FancyArrowPatch
from scipy.stats import chi2

from ctin_project.model.ctin_model_gru import CTINModel  # must output (vel, logstd)

# =========================
# Config
# =========================
csv_path   = "../euroc_output/ctin_mh01.csv"
model_path = "../ctin_model_EUROC_GRU_3D_NLL.pth"
window_size = 200
stride      = 5
batch_size  = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalization safety
STD_FLOOR = 1e-3
IMU_CLIP  = 80.0
Z_CLIP    = 10.0

# Colors (Okabe–Ito)
COL_GT   = "#000000"
COL_CTIN = "#E69F00"
COL_OV2  = "#56B4E9"
COL_SCAT = "#0072B2"
COL_CONE = "#D55E00"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "savefig.dpi": 300
})

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
    """ vel [N,D], t [N] -> pos [N,D], starting at 0 """
    N = vel.shape[0]
    pos = np.zeros_like(vel, dtype=np.float64)
    if N <= 1: return pos
    dt = np.diff(t).astype(np.float64)
    dt = np.clip(dt, 1e-4, 1.0)
    incr = 0.5 * (vel[:-1] + vel[1:]) * dt[:, None]
    pos[1:] = np.cumsum(incr, axis=0)
    return pos

def trapezoid_var_propagate(var_v, t):
    """
    Propagate diagonal velocity variances to position variances under
    trapezoidal integration, assuming independence across time & axes.
      var_v: [N,D]  per-axis velocity variance at center timestamps
      t:     [N]    seconds
    Returns: pos_var [N,D]
    """
    N, D = var_v.shape
    pos_var = np.zeros_like(var_v, dtype=np.float64)
    if N <= 1: return pos_var
    dt = np.diff(t).astype(np.float64)
    dt = np.clip(dt, 1e-4, 1.0)
    # var(0.5*(v_{k-1}+v_k)*dt) = (0.5*dt)^2 * (var_{k-1} + var_k)
    incr_var = (0.5 * dt[:, None])**2 * (var_v[:-1] + var_v[1:])
    pos_var[1:] = np.cumsum(incr_var, axis=0)
    return pos_var

def _arrow(ax, P, color):
    if len(P) < 2: return
    a = FancyArrowPatch((P[-2,0], P[-2,1]), (P[-1,0], P[-1,1]),
                        arrowstyle="-|>", mutation_scale=10, lw=0, color=color, zorder=6)
    ax.add_patch(a)

def align_se3(P_hat, P):
    """
    Rigid alignment (SE(3)) of predicted P_hat to GT P.
    Inputs: P_hat [N,3], P [N,3]
    Returns: P_hat_aligned, R, t
    """
    # centroids
    c_hat = P_hat.mean(axis=0, keepdims=True)
    c_gt  = P.mean(axis=0,  keepdims=True)
    X = P_hat - c_hat
    Y = P - c_gt
    # SVD for rotation
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # translation
    t = (c_gt - c_hat @ R.T).reshape(1, 3)
    # aligned
    P_hat_aligned = (P_hat @ R.T) + t
    return P_hat_aligned, R, t

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
# Normalize IMU (per file)
# =========================
imu = np.clip(imu, -IMU_CLIP, IMU_CLIP)
m = imu.mean(axis=0, dtype=np.float64)
s = np.maximum(imu.std(axis=0, dtype=np.float64), STD_FLOOR)
imu = (imu - m.astype(np.float32)) / s.astype(np.float32)
imu = np.clip(imu, -Z_CLIP, Z_CLIP)

# =========================
# Windows & center indices
# =========================
X_windows = create_windows(imu, window_size, stride)
if X_windows.shape[0] == 0:
    raise RuntimeError("No windows created; check window_size/stride vs file length.")

Nw = X_windows.shape[0]
mid = window_size // 2
center_idx = np.arange(mid, mid + stride*Nw, stride)
assert center_idx[-1] < len(df), "Center indices exceed dataframe length."
t_center = t_all[center_idx]

# =========================
# Inference (read vel and log-σ)
# =========================
model = CTINModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

vel_list, logstd_list = [], []
with torch.no_grad():
    X = torch.tensor(X_windows, dtype=torch.float32, device=device)
    X = torch.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
    for i in range(0, X.shape[0], batch_size):
        pv, pl = model(X[i:i+batch_size])          # pv/pl: [B,T,3]
        vel_list.append(pv[:, mid, :].detach().cpu())
        logstd_list.append(pl[:, mid, :].detach().cpu())

center_pred = torch.cat(vel_list, dim=0)           # [N,3]
center_logstd = torch.cat(logstd_list, dim=0)      # [N,3]
sigma = center_logstd.exp().numpy().astype(np.float64)  # [N,3]
var_v = sigma**2                                        # [N,3]

# Optional de-bias (velocity mean)
center_pred = center_pred - center_pred.mean(dim=0, keepdim=True)

# =========================
# Positions & position variance
# =========================
vel_pred = center_pred.numpy().astype(np.float64)
pos_pred = trapezoid_integrate(vel_pred, t_center)         # [N,3]
pos_var  = trapezoid_var_propagate(var_v, t_center)        # [N,3]
pos_std  = np.sqrt(np.clip(pos_var, 0.0, None))            # [N,3]

# =========================
# Ground truth positions
# =========================
if has_gt_pos:
    pos_gt_all = df[['gt_x','gt_y','gt_z']].to_numpy(dtype=np.float64)
    pos_gt = pos_gt_all[center_idx] - pos_gt_all[center_idx[0]]
else:
    v_gt_c = vel_gt[center_idx]
    pos_gt = trapezoid_integrate(v_gt_c, t_center)

# =========================
# SE(3) alignment for trajectory metrics
# =========================
PR = pos_pred.copy()
GT = pos_gt.copy()
PR_aligned, R_star, t_star = align_se3(PR, GT)

# =========================
# Metrics (ALIGNED)
# =========================
err_vec = PR_aligned - GT                     # aligned error vectors [N,3]
err_norm = np.linalg.norm(err_vec, axis=1)    # 3D norm
err_h    = np.linalg.norm(err_vec[:, :2], axis=1)  # horizontal norm
err_v    = np.abs(err_vec[:, 2])

ATE_mean  = float(err_norm.mean())                 # mean abs error
ATE_med   = float(np.median(err_norm))
ATE_p95   = float(np.percentile(err_norm, 95))
ATE_RMSE  = float(np.sqrt(np.mean(err_norm**2)))   # true ATE-RMSE
RMSE      = float(np.sqrt(mean_squared_error(GT, PR_aligned)))  # equals ATE_RMSE
Emax      = float(err_norm.max())

print("\nCTIN Evaluation (center-cadence, aligned):")
print(f"ATE mean    : {ATE_mean:.3f} m")
print(f"ATE median  : {ATE_med:.3f} m")
print(f"ATE p95     : {ATE_p95:.3f} m")
print(f"ATE RMSE    : {ATE_RMSE:.3f} m")
print(f"Max error   : {Emax:.3f} m")

# =========================
# Calibration: NEES (2D horizontal) coverage
# =========================
# Diagonal Σ_xy from propagated variances (horizontal only)
Sigma_xy = np.stack([pos_var[:,0], pos_var[:,1]], axis=-1)  # [N,2]
Sigma_xy = np.clip(Sigma_xy, 1e-12, None)

err_h_vec = err_vec[:, :2]
nees = (err_h_vec[:,0]**2 / Sigma_xy[:,0]) + (err_h_vec[:,1]**2 / Sigma_xy[:,1])  # [N]

thr95 = chi2.ppf(0.95, df=2)     # 5.991
thr997 = chi2.ppf(0.997, df=2)   # 11.829

coverage95  = 100.0 * np.mean(nees <= thr95)
coverage3sg = 100.0 * np.mean(nees <= thr997)
mean_nees   = float(nees.mean())

print(f"NEES mean (2D): {mean_nees:.2f} (target ≈ dof=2)")
print(f"NEES coverage @95%: {coverage95:.1f}% (target ~95%)")
print(f"NEES coverage @3σ:  {coverage3sg:.1f}% (target ~99.7%)")

# =========================
# Plots
# =========================
GT0 = GT - GT[0]
PR0 = PR_aligned - PR_aligned[0]
t_rel = t_center - t_center[0]

fig = plt.figure(figsize=(7.2, 5.4))
fig.suptitle("EuRoC Dataset (UAV)", y=0.98, fontsize=14, fontweight="bold")

# (1) CDF of ATE (aligned)
ax1 = fig.add_subplot(2,2,1)
s = np.sort(err_norm); y = np.linspace(0,1,len(s))
ax1.plot(s, y, color=COL_CTIN, lw=2, label="CTIN")
p50 = float(np.percentile(err_norm, 50))
p95 = ATE_p95
ax1.axvline(p50, color=COL_CTIN, ls=":", lw=1)
ax1.text(p50, 0.05, f"median={p50:.2f} m", rotation=90, va="bottom", ha="right", fontsize=8)
ax1.axvline(p95, color="k", ls="--", lw=1)
ax1.text(p95, 0.1, f"p95={p95:.2f} m", rotation=90, va="bottom", ha="right", fontsize=8)
ax1.set_xlabel("ATE [m]"); ax1.set_ylabel("Cumulative probability")
ax1.set_title("CDF of ATE (aligned)"); ax1.grid(True, alpha=0.3)
ax1.legend(frameon=False, loc="lower left")

# (2) Error vs Time + a visual 95% radius band (diag approx for plotting)
ax2 = fig.add_subplot(2,2,2)
ax2.plot(t_rel, err_h, color=COL_CTIN, lw=2, label="Horizontal error |e_h|")
# visual band using sqrt(trace(Σ_xy)) — for intuition only (official coverage via NEES above)
rad95_vis = np.sqrt(chi2.ppf(0.95, df=2)) * np.sqrt(pos_var[:,0] + pos_var[:,1])
ax2.fill_between(t_rel, 0, rad95_vis, alpha=0.15, label="Visual 95% band (XY)")
ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Error [m]")
ax2.set_title(f"Error vs Time  — NEES 95% coverage: {coverage95:.1f}%")
ax2.grid(True, alpha=0.3); ax2.legend(frameon=False, loc="upper right")

# (3) Top-down Trajectory (XY, aligned)
ax3 = fig.add_subplot(2,2,3)
ax3.plot(GT0[:,0], GT0[:,1], color=COL_GT,   lw=2.2, label="GT")
ax3.plot(PR0[:,0], PR0[:,1], color=COL_CTIN, lw=2.0, ls="--", label="CTIN (aligned)")
ax3.scatter([GT0[0,0]],[GT0[0,1]], c=COL_GT,   s=24, marker="o")
ax3.scatter([PR0[0,0]],[PR0[0,1]], c=COL_CTIN, s=22, marker="o")
ax3.scatter([GT0[-1,0]],[GT0[-1,1]], c=COL_GT,   s=30, marker="X")
ax3.scatter([PR0[-1,0]],[PR0[-1,1]], c=COL_CTIN, s=28, marker="X")
_arrow(ax3, PR0, COL_CTIN)
ax3.set_aspect("equal", adjustable="box"); ax3.margins(0.03)
ax3.set_xlabel("X [m]"); ax3.set_ylabel("Y [m]")
ax3.set_title("Top-down Trajectory (XY, aligned)")
ax3.grid(True, alpha=0.25); ax3.legend(frameon=False, loc="lower left")

# (4) RMSE bar (dataset label auto from path)
ax4 = fig.add_subplot(2,2,4)
seq_name = os.path.splitext(os.path.basename(csv_path))[0]  # e.g., ctin_mh01
labels = ["CTIN", "OV²SLAM-Fast\n(200 Hz)"]
ov2slam_rmse = 0.192   # <-- replace with the right number for THIS sequence if you have it
vals   = [float(RMSE), ov2slam_rmse]
colors = [COL_CTIN, COL_OV2]
bars = ax4.bar(np.arange(2), vals, color=colors, alpha=0.95)
for b, v in zip(bars, vals):
    ax4.text(b.get_x()+b.get_width()/2, v + 0.02*max(vals), f"{v:.3f}",
             ha="center", va="bottom", fontsize=8)
ax4.set_xticks(np.arange(2)); ax4.set_xticklabels(labels)
ax4.set_ylabel("ATE RMSE [m]")
ax4.set_title(f"RMSE Comparison ({seq_name})")
ax4.grid(True, axis="y", alpha=0.3)

fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("Euroc/euroc_uav_2x2_with_cov_aligned.png")
fig.savefig("euroc_uav_2x2_with_cov_aligned.svg")
plt.show()

# =========================
# Velocity bands per axis (95%)
# =========================
fig2, axs = plt.subplots(3, 1, figsize=(7.2, 5.0), sharex=True)
names = ["vx", "vy", "vz"]
for d in range(3):
    mu = vel_pred[:, d]
    std = np.sqrt(np.clip(var_v[:, d], 0.0, None))
    axs[d].plot(t_rel, mu, color=COL_CTIN, lw=1.5, label=f"pred {names[d]}")
    axs[d].fill_between(t_rel, mu - 1.96*std, mu + 1.96*std, alpha=0.2, label="±1.96σ")
    axs[d].set_ylabel(f"{names[d]} [m/s]")
    axs[d].grid(True, alpha=0.3)
axs[-1].set_xlabel("Time [s]")
axs[0].set_title("Predicted Velocity with 95% Bands")
axs[0].legend(frameon=False, loc="upper right")
fig2.tight_layout()
fig2.savefig("euroc_velocity_bands.png", dpi=300)
plt.show()

# =========================
# Velocity 3σ "cone" scatter (per-axis)
# =========================
def cone_violation_rate(err_axis, sigma_axis, k=3.0):
    """Percent of points outside ±kσ: i.e., sigma < |err|/k."""
    return 100.0 * np.mean(sigma_axis < (np.abs(err_axis) / k))

def coverage_stats(err_axis, sigma_axis):
    z = np.abs(err_axis) / np.clip(sigma_axis, 1e-12, None)
    return (float(np.mean(z <= 1.0)),
            float(np.mean(z <= 2.0)),
            float(np.mean(z <= 3.0)))

vel_gt_c = vel_gt[center_idx].astype(np.float64)   # [N,3]
vel_err  = vel_pred - vel_gt_c                     # [N,3]
vel_std  = np.sqrt(np.clip(var_v, 0.0, None))      # [N,3]

print("\n=== Velocity calibration (|e_v| / σ_v) coverage ===")
AX_LABS  = ["Vx", "Vy", "Vz"]
for i, lab in enumerate(AX_LABS):
    c1, c2, c3 = coverage_stats(vel_err[:, i], vel_std[:, i])
    print(f"{lab}: ≤1σ {c1*100:5.1f}% | ≤2σ {c2*100:5.1f}% | ≤3σ {c3*100:5.1f}%")

fig, axs = plt.subplots(3, 1, figsize=(4.0, 9.6), sharex=False)
fig.suptitle("Velocity Uncertainty vs Error — 3σ Cone Check", y=0.995, fontsize=12, fontweight="bold")

for i, ax in enumerate(axs):
    e = vel_err[:, i]         # signed error (m/s)
    s = vel_std[:, i]         # predicted σ (m/s)

    xmax = np.percentile(np.abs(e), 99.5) if e.size else 1.0
    xs = np.linspace(-xmax, +xmax, 400)

    # cone boundary and over-confident region (below the cone)
    ax.plot(xs, np.abs(xs) / 3.0, ls="--", lw=1.2, color=COL_CONE)
    ax.fill_between(xs, 0.0, np.abs(xs) / 3.0, alpha=0.10, color=COL_CONE)

    ax.scatter(e, s, s=6, alpha=0.35, color=COL_SCAT)

    vr = cone_violation_rate(e, s, k=3.0)
    ax.set_title(f"{AX_LABS[i]}  •  outside 3σ: {vr:.1f}%")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(bottom=0)

    if i == 2:
        ax.set_xlabel("Error (m/s)")
    ax.set_ylabel(r"Predicted $\sigma$ (m/s)")

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("euroc_vel_sigma_vs_error_3sigma_cone_vertical.png", dpi=300, bbox_inches="tight")
fig.savefig("euroc_vel_sigma_vs_error_3sigma_cone_vertical.svg", bbox_inches="tight")
plt.show()
