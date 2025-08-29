import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pathlib import Path

# ---------------- Config ----------------
csv_path   = "ctin_dataset_137102747096458.csv"
model_path = "../ctin_model_tlio_GRU_highStride.pth"
window_size = 200
stride      = 10
batch_size  = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For timestamp construction when CSV has no 'timestamp' column:
FALLBACK_HZ = 100.0  # TLIO_golden is commonly 100 Hz

# --- choose your old-code dt (your previous script used dt=0.05)
OLD_DT = 0.05

# Poster-safe colors (Okabe–Ito)
COL_GT   = "#000000"  # black
COL_OLD  = "#0072B2"  # blue (Old)
COL_NEW  = "#E69F00"  # orange (New)

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "savefig.dpi": 300
})

# ---------------- Helpers ----------------
def create_windows(data, T, S):
    n = len(data)
    idxs = range(0, n - T + 1, S)
    out = [data[i:i+T] for i in idxs]
    return np.stack(out) if out else np.zeros((0, T, data.shape[1]), dtype=data.dtype)

def integrate_riemann_const_dt(vel, dt):
    """Old method: simple left Riemann sum with constant dt."""
    pos = np.zeros_like(vel)
    if len(vel) == 0: return pos
    pos[0] = vel[0] * dt
    for t in range(1, vel.shape[0]):
        pos[t] = pos[t-1] + vel[t-1] * dt
    return pos

def trapezoid_integrate(vel, t):
    """New method: timestamp-aware trapezoidal integration."""
    N = len(vel)
    pos = np.zeros_like(vel)
    if N <= 1: return pos
    dt = np.diff(t).astype(np.float64)
    # guard against weird gaps
    dt = np.clip(dt, 1e-4, 10.0)
    incr = 0.5 * (vel[:-1] + vel[1:]) * dt[:, None]
    pos[1:] = np.cumsum(incr, axis=0)
    return pos

def metrics_xy(P, GT):
    e = P - GT
    e_norm = np.linalg.norm(e, axis=1)
    return dict(
        mean=float(e_norm.mean()),
        p95=float(np.percentile(e_norm, 95)),
        rmse=float(np.sqrt(mean_squared_error(GT, P))),
        max=float(e_norm.max()),
        series=e_norm
    )

# ---------------- Load CSV ----------------
df = pd.read_csv(csv_path).dropna()

imu = df[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']].to_numpy(dtype=np.float32)

# GT velocity (for fallback GT integration if position not present)
has_gt_pos = False
pos_gt_all = None
vel_gt = None

# Optional: GT positions if available
for cand in [('gt_x','gt_y'), ('pos_x','pos_y')]:
    if set(cand).issubset(df.columns):
        pos_gt_all = df[list(cand)].to_numpy(dtype=np.float64)
        has_gt_pos = True
        break

if not has_gt_pos:
    # then we need GT velocities in CSV
    if set(['vel_x','vel_y']).issubset(df.columns):
        vel_gt = df[['vel_x','vel_y']].to_numpy(dtype=np.float64)
    else:
        raise RuntimeError("No GT position or velocity columns found for alignment.")

# Timestamps
if 'timestamp' in df.columns:
    t_all = df['timestamp'].to_numpy(dtype=np.float64)
else:
    t0 = 0.0
    dt_nom = 1.0 / FALLBACK_HZ
    t_all = t0 + np.arange(len(df), dtype=np.float64) * dt_nom

# ---------------- Normalize IMU (per file) ----------------
m = imu.mean(axis=0, dtype=np.float64)
s = imu.std(axis=0, dtype=np.float64) + 1e-6
imu = (imu - m.astype(np.float32)) / s.astype(np.float32)

# ---------------- Windows & centers ----------------
X_windows = create_windows(imu, window_size, stride)
if X_windows.shape[0] == 0:
    raise RuntimeError("No windows created. Check window_size/stride vs sequence length.")
N = X_windows.shape[0]
mid = window_size // 2
center_idx = np.arange(mid, mid + stride*N, stride)
assert center_idx[-1] < len(df), "Center indices exceed dataframe length."

t_center = t_all[center_idx]                # timestamps at center predictions

# ---------------- Load CTIN model & predict center velocities ----------------
from ctin_project.model.ctin_model_gru import CTINModel
model = CTINModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

pred_chunks = []
with torch.no_grad():
    X = torch.tensor(X_windows, dtype=torch.float32, device=device)
    X = torch.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
    for i in range(0, X.shape[0], batch_size):
        pv, _ = model(X[i:i+batch_size])      # [B, T, 2]
        center = pv[:, mid, :]                # [B, 2]
        pred_chunks.append(center.cpu())
center_pred = torch.cat(pred_chunks, dim=0)   # [N, 2]
vel_pred_np = center_pred.numpy().astype(np.float64)

# ---------------- Old vs New positions ----------------
# OLD: your previous approach (constant dt Riemann sum)
pos_old = integrate_riemann_const_dt(vel_pred_np, dt=OLD_DT)

# NEW: de-bias + trapezoidal with real timestamps
vel_pred_db = vel_pred_np - vel_pred_np.mean(axis=0, keepdims=True)
pos_new = trapezoid_integrate(vel_pred_db, t_center)

# ---------------- Align GT positions to center indices ----------------
if has_gt_pos:
    pos_gt_aligned = pos_gt_all[center_idx] - pos_gt_all[center_idx[0]]
else:
    # Integrate GT velocity over t_center to get a fair GT position track
    v_gt_c = vel_gt[center_idx]
    pos_gt_aligned = trapezoid_integrate(v_gt_c, t_center)

# ---------------- Metrics ----------------
m_old = metrics_xy(pos_old, pos_gt_aligned)
m_new = metrics_xy(pos_new, pos_gt_aligned)

print("\n=== CTIN Inference Comparison (Old vs New) ===")
print("OLD  -> mean {:.3f}  p95 {:.3f}  RMSE {:.3f}  max {:.3f}"
      .format(m_old['mean'], m_old['p95'], m_old['rmse'], m_old['max']))
print("NEW  -> mean {:.3f}  p95 {:.3f}  RMSE {:.3f}  max {:.3f}"
      .format(m_new['mean'], m_new['p95'], m_new['rmse'], m_new['max']))

# ---------------- Save NPZs (optional, handy for overlays later) ----------------
outdir = Path("./compare_old_new_results")
outdir.mkdir(parents=True, exist_ok=True)
np.savez(outdir / "ctin_old.npz", t=t_center, pos_pred=pos_old, pos_gt=pos_gt_aligned)
np.savez(outdir / "ctin_new.npz", t=t_center, pos_pred=pos_new, pos_gt=pos_gt_aligned)

# ---------------- PPT-friendly individual plots ----------------
t_rel = t_center - t_center[0]

# 1) XY Overlay
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.plot(pos_gt_aligned[:,0], pos_gt_aligned[:,1], color=COL_GT,   lw=2.4, label="GT")
ax.plot(pos_old[:,0],        pos_old[:,1],        color=COL_OLD,  lw=2.2, ls="--", label="Old (Riemann)")
ax.plot(pos_new[:,0],        pos_new[:,1],        color=COL_NEW,  lw=2.2, ls="-",  label="New (De-biased + Trapz)")
ax.set_aspect("equal", adjustable="box"); ax.margins(0.03)
ax.grid(True, alpha=0.25, linewidth=0.8)
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.set_title("Trajectory Overlay — GT vs Old vs New")
ax.legend(frameon=False, loc="best")
plt.tight_layout(); plt.savefig(outdir / "overlay_old_vs_new.png", dpi=300); plt.show()

# 2) Error vs Time
plt.figure(figsize=(5.6, 4.2))
plt.plot(t_rel, m_old['series'], color=COL_OLD, lw=2, ls="--", label="Old")
plt.plot(t_rel, m_new['series'], color=COL_NEW, lw=2, label="New")
plt.xlabel("Time [s]"); plt.ylabel("Error [m]")
plt.title("Error vs Time"); plt.grid(True, alpha=0.3)
plt.legend(frameon=False, loc="upper right")
plt.tight_layout(); plt.savefig(outdir / "error_vs_time_old_vs_new.png", dpi=300); plt.show()

# 3) CDF of ATE
def cdf_plot(series, color, label):
    s = np.sort(series); y = np.linspace(0,1,len(s))
    plt.plot(s, y, color=color, lw=2, label=label)
plt.figure(figsize=(5.2,4.2))
cdf_plot(m_old['series'], COL_OLD, "Old")
cdf_plot(m_new['series'], COL_NEW, "New")
plt.xlabel("ATE [m]"); plt.ylabel("Cumulative probability")
plt.title("CDF of Absolute Trajectory Error"); plt.grid(True, alpha=0.3)
plt.legend(frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig(outdir / "cdf_old_vs_new.png", dpi=300); plt.show()

# 4) Metrics Bar Chart
labels = ["ATE mean", "ATE p95", "RMSE", "Max error"]
vals_old = [m_old['mean'], m_old['p95'], m_old['rmse'], m_old['max']]
vals_new = [m_new['mean'], m_new['p95'], m_new['rmse'], m_new['max']]
x = np.arange(len(labels)); w = 0.42
plt.figure(figsize=(6.0, 4.2))
b1 = plt.bar(x - w/2, vals_old, width=w, color=COL_OLD, alpha=0.95, label="Old")
b2 = plt.bar(x + w/2, vals_new, width=w, color=COL_NEW, alpha=0.95, label="New")
ymax = max(max(vals_old), max(vals_new))
for bars in (b1, b2):
    for b in bars:
        v = b.get_height()
        plt.text(b.get_x()+b.get_width()/2, v + 0.02*ymax, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
plt.xticks(x, labels); plt.ylabel("Error [m]")
plt.title("Metrics: Old vs New Inference")
plt.grid(True, axis="y", alpha=0.3)
plt.legend(frameon=False, loc="upper right")
plt.tight_layout(); plt.savefig(outdir / "metrics_old_vs_new.png", dpi=300); plt.show()

print(f"Saved figures to: {outdir.resolve()}")
