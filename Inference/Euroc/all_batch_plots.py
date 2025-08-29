# infer_euroc_batch_all_plots.py
import os, glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.metrics import mean_squared_error
from scipy.stats import chi2

from ctin_project.model.ctin_model_gru import CTINModel  # must output (vel, logstd)

# =========================
# Config
# =========================
pattern     = "../../euroc_output/*.csv"
model_path  = "../../ctin_model_EUROC_GRU_3D_NLL.pth"
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
COL_SCAT = "#0072B2"
COL_CONE = "#D55E00"

# Output dirs
OUT_DIR = "./Euroc/batch_plots"
os.makedirs(OUT_DIR, exist_ok=True)

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
    Propagate diagonal velocity variances to position variances under trapezoidal
    integration, assuming independence across time & axes.
      var_v: [N,D] at center timestamps
      t:     [N]
    """
    N, D = var_v.shape
    pos_var = np.zeros_like(var_v, dtype=np.float64)
    if N <= 1: return pos_var
    dt = np.diff(t).astype(np.float64)
    dt = np.clip(dt, 1e-4, 1.0)
    incr_var = (0.5 * dt[:, None])**2 * (var_v[:-1] + var_v[1:])
    pos_var[1:] = np.cumsum(incr_var, axis=0)
    return pos_var

def align_se3(P_hat, P):
    """ Rigid alignment (SE(3)) of predicted P_hat to GT P. """
    c_hat = P_hat.mean(axis=0, keepdims=True)
    c_gt  = P.mean(axis=0,  keepdims=True)
    X = P_hat - c_hat
    Y = P - c_gt
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = (c_gt - c_hat @ R.T).reshape(1, 3)
    P_hat_aligned = (P_hat @ R.T) + t
    return P_hat_aligned, R, t

def _arrow(ax, P, color):
    if len(P) < 2: return
    a = FancyArrowPatch((P[-2,0], P[-2,1]), (P[-1,0], P[-1,1]),
                        arrowstyle="-|>", mutation_scale=10, lw=0, color=color, zorder=6)
    ax.add_patch(a)

def save_topdown(seq, GT, PR_aligned):
    GT0 = GT - GT[0]
    PR0 = PR_aligned - PR_aligned[0]
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(GT0[:,0], GT0[:,1], color=COL_GT,   lw=2.2, label="GT")
    ax.plot(PR0[:,0], PR0[:,1], color=COL_CTIN, lw=2.0, ls="--", label="CTIN (aligned)")
    ax.scatter([GT0[0,0]],[GT0[0,1]], c=COL_GT,   s=24, marker="o")
    ax.scatter([PR0[0,0]],[PR0[0,1]], c=COL_CTIN, s=22, marker="o")
    ax.scatter([GT0[-1,0]],[GT0[-1,1]], c=COL_GT,   s=32, marker="X")
    ax.scatter([PR0[-1,0]],[PR0[-1,1]], c=COL_CTIN, s=30, marker="X")
    _arrow(ax, PR0, COL_CTIN)
    ax.set_aspect("equal", "box"); ax.margins(0.03)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title(f"{seq} — Top-down (XY, aligned)")
    ax.grid(True, alpha=0.25); ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"{seq}_topdown.png")
    fig.savefig(p); plt.close(fig)
    return p

def save_error_vs_time(seq, t_rel, err_h, pos_var=None):
    fig = plt.figure(figsize=(6.0, 3.0))
    ax = fig.add_subplot(1,1,1)
    ax.plot(t_rel, err_h, color=COL_CTIN, lw=1.8, label="|e_xy|")
    if pos_var is not None:
        rad95_vis = np.sqrt(chi2.ppf(0.95, df=2)) * np.sqrt(np.clip(pos_var[:,0] + pos_var[:,1], 0, None))
        ax.fill_between(t_rel, 0, rad95_vis, alpha=0.15, label="Visual 95% band (XY)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Error [m]")
    ax.set_title(f"{seq} — Error vs Time")
    ax.grid(True, alpha=0.3); ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"{seq}_error_vs_time.png")
    fig.savefig(p); plt.close(fig)
    return p

def save_cdf(seq, err_norm):
    s = np.sort(err_norm); y = np.linspace(0,1,len(s))
    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(s, y, color=COL_CTIN, lw=2)
    ax.set_xlabel("ATE [m]"); ax.set_ylabel("CDF")
    ax.set_title(f"{seq} — CDF of ATE (aligned)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"{seq}_cdf_ate.png")
    fig.savefig(p); plt.close(fig)
    return p

def save_vel_cone(seq, vel_err, vel_std):
    AX_LABS = ["Vx", "Vy", "Vz"]
    fig, axs = plt.subplots(3, 1, figsize=(4.0, 9.6), sharex=False)
    fig.suptitle(f"{seq} — Velocity 3σ Cone", y=0.995, fontsize=12, fontweight="bold")
    for i, ax in enumerate(axs):
        e = vel_err[:, i]
        s = vel_std[:, i]
        xmax = np.percentile(np.abs(e), 99.5) if e.size else 1.0
        xs = np.linspace(-xmax, +xmax, 400)
        ax.plot(xs, np.abs(xs) / 3.0, ls="--", lw=1.2, color=COL_CONE)
        ax.fill_between(xs, 0.0, np.abs(xs) / 3.0, alpha=0.10, color=COL_CONE)
        ax.scatter(e, s, s=6, alpha=0.35, color=COL_SCAT)
        ax.set_title(AX_LABS[i]); ax.grid(True, alpha=0.3)
        ax.set_xlim(-xmax, xmax); ax.set_ylim(bottom=0)
        if i == 2: ax.set_xlabel("Error (m/s)")
        ax.set_ylabel(r"$\sigma$ (m/s)")
    fig.tight_layout(rect=[0,0,1,0.97])
    p = os.path.join(OUT_DIR, f"{seq}_vel_cone.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return p

def coverage_stats(err_axis, sigma_axis):
    z = np.abs(err_axis) / np.clip(sigma_axis, 1e-12, None)
    return (float(np.mean(z <= 1.0)),
            float(np.mean(z <= 2.0)),
            float(np.mean(z <= 3.0)))

# =========================
# Model
# =========================
model = CTINModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =========================
# Batch over CSVs
# =========================
csv_files = sorted(glob.glob(pattern))
if not csv_files:
    raise FileNotFoundError(f"No files match: {pattern}")
print("[FILES]", *[os.path.basename(f) for f in csv_files], sep="\n  ")

summary = []

for csv_path in csv_files:
    seq = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n[RUN] {seq}")

    # --- Load CSV ---
    df = pd.read_csv(csv_path).dropna()
    for c in ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','vx','vy','vz','timestamp']:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c} in {seq}")

    has_gt_pos = set(['gt_x','gt_y','gt_z']).issubset(df.columns)

    imu    = df[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']].to_numpy(dtype=np.float32)
    vel_gt = df[['vx','vy','vz']].to_numpy(dtype=np.float64)
    t_all  = df['timestamp'].to_numpy(dtype=np.float64)

    # --- Normalize IMU (per-file) ---
    imu = np.clip(imu, -IMU_CLIP, IMU_CLIP)
    m = imu.mean(axis=0, dtype=np.float64)
    s = np.maximum(imu.std(axis=0, dtype=np.float64), STD_FLOOR)
    imu = (imu - m.astype(np.float32)) / s.astype(np.float32)
    imu = np.clip(imu, -Z_CLIP, Z_CLIP)

    # --- Windows (center cadence) ---
    Xw = create_windows(imu, window_size, stride)
    if Xw.shape[0] == 0:
        print(f"[SKIP] no windows for {seq}")
        continue
    mid = window_size // 2
    Nw  = Xw.shape[0]
    center_idx = np.arange(mid, mid + stride*Nw, stride)
    t_center   = t_all[center_idx]

    # --- Inference: velocity mean & log-σ (center frames) ---
    vel_list, logstd_list = [], []
    with torch.no_grad():
        X = torch.tensor(Xw, dtype=torch.float32, device=device)
        X = torch.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
        for i in range(0, X.shape[0], batch_size):
            pv, pl = model(X[i:i+batch_size])          # pv/pl: [B,T,3]
            vel_list.append(pv[:, mid, :].detach().cpu())
            logstd_list.append(pl[:, mid, :].detach().cpu())

    vel_pred = torch.cat(vel_list, dim=0).numpy().astype(np.float64)      # [N,3]
    logstd   = torch.cat(logstd_list, dim=0).numpy().astype(np.float64)   # [N,3]
    sigma_v  = np.exp(logstd)                                             # [N,3] m/s
    var_v    = sigma_v**2

    # Optional de-bias (velocity mean)
    vel_pred = vel_pred - vel_pred.mean(axis=0, keepdims=True)

    # --- Positions & position variance (center cadence) ---
    pos_pred = trapezoid_integrate(vel_pred, t_center)              # [N,3]
    pos_var  = trapezoid_var_propagate(var_v, t_center)             # [N,3]
    pos_std  = np.sqrt(np.clip(pos_var, 0.0, None))

    # --- Ground truth positions ---
    if has_gt_pos:
        pos_gt_all = df[['gt_x','gt_y','gt_z']].to_numpy(dtype=np.float64)
        pos_gt = pos_gt_all[center_idx] - pos_gt_all[center_idx[0]]
    else:
        v_gt_c = vel_gt[center_idx]
        pos_gt = trapezoid_integrate(v_gt_c, t_center)

    # --- SE(3) alignment ---
    PR_al, R_star, t_star = align_se3(pos_pred.copy(), pos_gt.copy())

    # --- Errors (aligned) ---
    err_vec = PR_al - pos_gt
    err_norm = np.linalg.norm(err_vec, axis=1)
    err_h    = np.linalg.norm(err_vec[:, :2], axis=1)

    # --- Velocity error & std (for cone) at center cadence ---
    vel_gt_c = vel_gt[center_idx].astype(np.float64)
    vel_err  = vel_pred - vel_gt_c
    vel_std  = np.sqrt(np.clip(var_v, 0.0, None))

    # --- Metrics ---
    ATE_mean  = float(err_norm.mean())
    ATE_med   = float(np.median(err_norm))
    ATE_p95   = float(np.percentile(err_norm, 95))
    ATE_RMSE  = float(np.sqrt(np.mean(err_norm**2)))
    Emax      = float(err_norm.max())

    # NEES coverage (horizontal, diag Σ)
    Sigma_xy = np.stack([pos_var[:,0], pos_var[:,1]], axis=-1)
    Sigma_xy = np.clip(Sigma_xy, 1e-12, None)
    nees = (err_vec[:,0]**2 / Sigma_xy[:,0]) + (err_vec[:,1]**2 / Sigma_xy[:,1])
    thr95, thr997 = chi2.ppf(0.95, 2), chi2.ppf(0.997, 2)
    coverage95  = 100.0 * np.mean(nees <= thr95)
    coverage3sg = 100.0 * np.mean(nees <= thr997)
    mean_nees   = float(nees.mean())

    # --- Save plots ---
    _ = save_topdown(seq, pos_gt, PR_al)
    _ = save_error_vs_time(seq, t_center - t_center[0], err_h, pos_var[:, :2])
    _ = save_cdf(seq, err_norm)
    _ = save_vel_cone(seq, vel_err, vel_std)

    # --- Console coverage for velocity (nice to see) ---
    c1x, c2x, c3x = coverage_stats(vel_err[:,0], vel_std[:,0])
    c1y, c2y, c3y = coverage_stats(vel_err[:,1], vel_std[:,1])
    c1z, c2z, c3z = coverage_stats(vel_err[:,2], vel_std[:,2])
    print(f"  ATE RMSE: {ATE_RMSE:.3f} m | p95: {ATE_p95:.3f} m")
    print(f"  NEES mean: {mean_nees:.2f} | cov95: {coverage95:.1f}% | cov3σ: {coverage3sg:.1f}%")
    print(f"  Vel cov ≤3σ: Vx {c3x*100:.1f}% | Vy {c3y*100:.1f}% | Vz {c3z*100:.1f}%")

    summary.append({
        "seq": seq,
        "ATE_mean_m": ATE_mean,
        "ATE_med_m":  ATE_med,
        "ATE_p95_m":  ATE_p95,
        "ATE_RMSE_m": ATE_RMSE,
        "ATE_max_m":  Emax,
        "NEES_mean":  mean_nees,
        "NEES_cov95_%": coverage95,
        "NEES_cov3sigma_%": coverage3sg,
        "Vel_cov3σ_Vx_%": c3x*100,
        "Vel_cov3σ_Vy_%": c3y*100,
        "Vel_cov3σ_Vz_%": c3z*100,
    })

# =========================
# Summary CSV
# =========================
if summary:
    df_sum = pd.DataFrame(summary).sort_values("seq")
    out_csv = os.path.join(OUT_DIR, "summary_metrics.csv")
    df_sum.to_csv(out_csv, index=False)
    print(f"\n[OK] Wrote summary: {out_csv}")
else:
    print("\n[WARN] Nothing processed.")
