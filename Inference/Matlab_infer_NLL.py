# infer_fused_ctin_quickfix_rmse_ate.py
import os, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ctin_project.model.ctin_model import CTINModel

# ===== Config =====
DATA_ROOT  = "../Datasets"
IMU_CSV    = os.path.join(DATA_ROOT, "imu_matlab", "imu_ctin_{SEQ}.csv")
GT_CSV     = os.path.join(DATA_ROOT, "gt_matlab",  "gt_vel_ctin_{SEQ}.csv")
SEQ_NAME   = "circle"                      # <-- change per sequence
MODEL_PATH = "ctin_matlab_trainonly_GRU_NLL.pth"
NORM_JSON  = "ctin_matlab_norm.json"       # if missing, per-file z-norm

WINDOW = 200
STRIDE = 1
DT     = 1/100.0
VEL_SCALE = 40.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "./inference_quickfix"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== Helpers =====
def create_windows(X, W, S):
    n = len(X)
    return np.stack([X[i:i+W] for i in range(0, n-W+1, S)]) if n>=W else np.zeros((0,W,X.shape[1]), X.dtype)

def integrate_trap(vel, dt):
    pos = np.zeros_like(vel, dtype=np.float64)
    if vel.shape[0] <= 1: return pos
    incr = 0.5*(vel[:-1]+vel[1:]) * dt
    pos[1:] = np.cumsum(incr, axis=0)
    return pos

def load_norm(path):
    if not os.path.exists(path): return None
    with open(path, "r") as f: d = json.load(f)
    return (np.array(d["imu_mean"], np.float32),
            np.array(d["imu_std"],  np.float32),
            float(d["vel_scale"]),
            bool(d.get("swap_acc_first", True)))

def coverage(err, std):
    z = np.abs(err) / np.clip(std, 1e-12, None)
    return (np.mean(z<=1.0), np.mean(z<=2.0), np.mean(z<=3.0))

# ===== Load data =====
imu = pd.read_csv(IMU_CSV.format(SEQ=SEQ_NAME), header=None).values.astype(np.float32)  # [T,6]
vel_gt_ms = pd.read_csv(GT_CSV.format(SEQ=SEQ_NAME),  header=None).values.astype(np.float32)  # [T,2] m/s

# Normalize IMU
norm = load_norm(NORM_JSON)
if norm is not None:
    imu_mean, imu_std, vs, swap = norm
    if swap:  # training expected [acc, gyro]
        imu = imu[:, [3,4,5, 0,1,2]]
    imu_norm = (imu - imu_mean) / imu_std
else:
    imu_mean = imu.mean(axis=0)
    imu_std  = imu.std(axis=0) + 1e-6
    imu_norm = (imu - imu_mean) / imu_std

T = min(len(imu_norm), len(vel_gt_ms))
imu_norm  = imu_norm[:T]
vel_gt_ms = vel_gt_ms[:T]

# ===== Windowing =====
X_win = create_windows(imu_norm, WINDOW, STRIDE)          # [Nw, W, 6]
if X_win.shape[0] == 0: raise RuntimeError("Not enough samples for one window.")
X_t   = torch.tensor(X_win, dtype=torch.float32, device=DEVICE)

# ===== Model =====
model = CTINModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== Predict (windows) =====
pred_vel_win, logstd_win = [], []
with torch.no_grad():
    for i in range(0, len(X_t), 64):
        pv, pl = model(X_t[i:i+64])        # pv,pl: [B, W, 2] (pv in scaled units)
        pred_vel_win.append(pv.cpu())
        logstd_win.append(pl.cpu())
pred_vel_win = torch.cat(pred_vel_win, dim=0).numpy()     # [Nw, W, 2]
logstd_win   = torch.cat(logstd_win,   dim=0).numpy()     # [Nw, W, 2]

# ===== Unscale =====
pred_vel_win_ms = pred_vel_win * VEL_SCALE                # m/s
sigma_win_ms    = np.exp(logstd_win) * VEL_SCALE          # m/s
var_win_ms2     = np.clip(sigma_win_ms**2, 1e-12, None)   # (m/s)^2

# ===== Overlap–Add Fusion (simple mean) =====
fused_vel_ms = np.zeros((T, 2), dtype=np.float64)
count        = np.zeros((T, 1), dtype=np.float64)
for i in range(pred_vel_win_ms.shape[0]):
    s = i; e = i + WINDOW
    if e > T: break
    fused_vel_ms[s:e] += pred_vel_win_ms[i]
    count[s:e]        += 1.0
nz = count[:,0] > 0
fused_vel_ms[nz] /= count[nz]
fused_vel_ms[~nz] = 0.0

# ===== Integrate to positions (meters) =====
pos_gt  = integrate_trap(vel_gt_ms,  DT)
pos_pr  = integrate_trap(fused_vel_ms, DT)

# ===== Metrics: ATE-mean vs ATE-RMSE (trajectory RMSE) =====
err_vec  = pos_pr - pos_gt                    # [N,2] meters
err_norm = np.linalg.norm(err_vec, axis=1)    # per-step ATE (m)

ATE_mean   = float(err_norm.mean())                             # mean(|e|)
ATE_median = float(np.median(err_norm))
ATE_p95    = float(np.percentile(err_norm, 95))
ATE_RMSE   = float(np.sqrt(np.mean(np.sum(err_vec**2, axis=1))))  # == trajectory RMSE

# per-axis position RMSE (m)
RMSE_x, RMSE_y = np.sqrt(np.mean(err_vec**2, axis=0)).tolist()

# velocity RMSE (m/s)
vel_rmse_ms = np.sqrt(np.mean((fused_vel_ms - vel_gt_ms)**2, axis=0))

print(f"[{SEQ_NAME}] Traj RMSE (ATE-RMSE): {ATE_RMSE:.3f} m | ATE mean {ATE_mean:.3f} m | "
      f"p50 {ATE_median:.3f} m | p95 {ATE_p95:.3f} m")
print(f"[{SEQ_NAME}] Pos RMSE per-axis: Rx={RMSE_x:.3f} m  Ry={RMSE_y:.3f} m")
print(f"[{SEQ_NAME}] Vel RMSE: Vx={vel_rmse_ms[0]:.2f} m/s  Vy={vel_rmse_ms[1]:.2f} m/s")

# Save a small metrics CSV for your table
metrics_row = {
    "seq": SEQ_NAME,
    "traj_rmse_m": ATE_RMSE,
    "ate_mean_m": ATE_mean,
    "ate_p50_m": ATE_median,
    "ate_p95_m": ATE_p95,
    "rmse_x_m": RMSE_x,
    "rmse_y_m": RMSE_y,
    "vel_rmse_vx_ms": float(vel_rmse_ms[0]),
    "vel_rmse_vy_ms": float(vel_rmse_ms[1]),
}
import csv
csv_path = os.path.join(OUT_DIR, f"{SEQ_NAME}_metrics.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(metrics_row.keys()))
    w.writeheader(); w.writerow(metrics_row)
print(f"[{SEQ_NAME}] Saved metrics -> {csv_path}")

# ===== Error vs Time & CDF of ATE =====
t = np.arange(len(pos_gt)) * DT
err_xy = err_norm
plt.figure(figsize=(6,3.0))
plt.plot(t, err_xy, lw=1.8)
plt.xlabel("Time [s]"); plt.ylabel("‖pos error‖ [m]")
plt.title(f"{SEQ_NAME} — Error vs Time")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"{SEQ_NAME}_error_vs_time.png"), dpi=220)
plt.show()

s = np.sort(err_xy); y = np.linspace(0, 1, len(s))
plt.figure(figsize=(4.2,3.6))
plt.plot(s, y, lw=2)
plt.xlabel("ATE [m]"); plt.ylabel("CDF")
plt.title(f"{SEQ_NAME} — CDF of ATE")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"{SEQ_NAME}_cdf_ate.png"), dpi=220)
plt.show()

# ===== Post-hoc σ temperature scaling (unchanged) =====
fused_var_ms2 = np.zeros((T,2), dtype=np.float64)
var_count     = np.zeros((T,2), dtype=np.float64)
for i in range(var_win_ms2.shape[0]):
    s = i; e = i + WINDOW
    if e > T: break
    fused_var_ms2[s:e] += var_win_ms2[i]
    var_count[s:e]     += 1.0
nz2 = var_count > 0
fused_var_ms2[nz2] /= var_count[nz2]
fused_var_ms2[~nz2] = np.inf

fused_std_ms = np.sqrt(np.clip(fused_var_ms2, 0, None))
vel_err_ms   = fused_vel_ms - vel_gt_ms

tau_x = float(np.sqrt(np.mean((vel_err_ms[:,0] / np.clip(fused_std_ms[:,0], 1e-12, None))**2)))
tau_y = float(np.sqrt(np.mean((vel_err_ms[:,1] / np.clip(fused_std_ms[:,1], 1e-12, None))**2)))
tau_x = float(np.clip(tau_x, 1e-3, 1e3))
tau_y = float(np.clip(tau_y, 1e-3, 1e3))

fused_std_ms_cal = fused_std_ms.copy()
fused_std_ms_cal[:,0] *= tau_x
fused_std_ms_cal[:,1] *= tau_y

c1x,c2x,c3x = coverage(vel_err_ms[:,0], fused_std_ms[:,0])
c1y,c2y,c3y = coverage(vel_err_ms[:,1], fused_std_ms[:,1])
print(f"[{SEQ_NAME}] Uncalibrated coverage X: ≤1σ {100*c1x:.1f}% | ≤2σ {100*c2x:.1f}% | ≤3σ {100*c3x:.1f}%")
print(f"[{SEQ_NAME}] Uncalibrated coverage Y: ≤1σ {100*c1y:.1f}% | ≤2σ {100*c2y:.1f}% | ≤3σ {100*c3y:.1f}%")

c1x,c2x,c3x = coverage(vel_err_ms[:,0], fused_std_ms_cal[:,0])
c1y,c2y,c3y = coverage(vel_err_ms[:,1], fused_std_ms_cal[:,1])
print(f"[{SEQ_NAME}] Calibrated   coverage X: ≤1σ {100*c1x:.1f}% | ≤2σ {100*c2x:.1f}% | ≤3σ {100*c3x:.1f}%")
print(f"[{SEQ_NAME}] Calibrated   coverage Y: ≤1σ {100*c1y:.1f}% | ≤2σ {100*c2y:.1f}% | ≤3σ {100*c3y:.1f}%")
print(f"[{SEQ_NAME}] τ (temp scale): X={tau_x:.2f}, Y={tau_y:.2f}")

# ===== Trajectory plot (title shows RMSE clearly) =====
plt.figure(figsize=(5,5))
plt.plot(pos_gt[:,0], pos_gt[:,1], label="GT", lw=2)
plt.plot(pos_pr[:,0], pos_pr[:,1], '--', label="CTIN (fused)", lw=2)
plt.axis('equal'); plt.grid(True); plt.xlabel("X [m]"); plt.ylabel("Y [m]")
plt.title(f"{SEQ_NAME} — Traj RMSE {ATE_RMSE:.2f} m  |  ATE mean {ATE_mean:.2f} m")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"{SEQ_NAME}_traj_fused.png"), dpi=220)
plt.show()

# ===== 3σ cone (calibrated σ) =====
fig, axs = plt.subplots(2,1, figsize=(4.2,6.8))
for i, ax in enumerate(axs):
    e = vel_err_ms[:,i]; s_cal = fused_std_ms_cal[:,i]
    xmax = np.percentile(np.abs(e), 99.5) if e.size else 1.0
    xs = np.linspace(-xmax,  xmax,  400)
    ax.plot(xs, np.abs(xs)/3.0, ls="--", label="3σ boundary")
    ax.fill_between(xs, 0, np.abs(xs)/3.0, alpha=0.08)
    ax.scatter(e, s_cal, s=6, alpha=0.35, label="calib σ")
    ax.set_xlim(-xmax, xmax); ax.set_ylim(bottom=0)
    ax.grid(True); ax.set_ylabel("σ (m/s)")
    if i==1: ax.set_xlabel("Error (m/s)")
    ax.set_title(["Vx","Vy"][i] + " — 3σ cone")
    ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"{SEQ_NAME}_3sigma_cone_calibrated.png"), dpi=300)
plt.show()
