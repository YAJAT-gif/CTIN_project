# infer_euroc_topdown_batch.py
import os, glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from ctin_project.model.ctin_model_gru import CTINModel  # must output (vel, logstd)

# ========= Config =========
pattern = "../../euroc_output/*.csv"    # <- all your per-seq CSVs
model_path  = "../../ctin_model_EUROC_GRU_3D_NLL.pth"
window_size = 200
stride      = 5
batch_size  = 64
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalization safety
STD_FLOOR = 1e-3
IMU_CLIP  = 80.0
Z_CLIP    = 10.0

# Colors (Okabe–Ito)
COL_GT   = "#000000"
COL_CTIN = "#E69F00"

# Output
OUT_DIR = "./Euroc/topdown_plots"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "savefig.dpi": 300
})

# ========= Helpers =========
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

def save_topdown_plot(seq_name, GT, PR_aligned):
    GT0 = GT - GT[0]
    PR0 = PR_aligned - PR_aligned[0]

    fig = plt.figure(figsize=(4.5, 4.5))  # square, Word-friendly
    ax = fig.add_subplot(1,1,1)
    ax.plot(GT0[:,0], GT0[:,1], color=COL_GT,   lw=2.2, label="GT")
    ax.plot(PR0[:,0], PR0[:,1], color=COL_CTIN, lw=2.0, ls="--", label="CTIN (aligned)")
    ax.scatter([GT0[0,0]],[GT0[0,1]], c=COL_GT,   s=24, marker="o")
    ax.scatter([PR0[0,0]],[PR0[0,1]], c=COL_CTIN, s=22, marker="o")
    ax.scatter([GT0[-1,0]],[GT0[-1,1]], c=COL_GT,   s=32, marker="X")
    ax.scatter([PR0[-1,0]],[PR0[-1,1]], c=COL_CTIN, s=30, marker="X")
    _arrow(ax, PR0, COL_CTIN)

    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.03)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title(f"{seq_name} — Top-down (XY, aligned)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower left")

    png = os.path.join(OUT_DIR, f"{seq_name}_topdown.png")
    svg = os.path.join(OUT_DIR, f"{seq_name}_topdown.svg")
    fig.tight_layout()
    fig.savefig(png)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg

# ========= Model =========
model = CTINModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ========= Run all sequences =========
csv_files = sorted(glob.glob(pattern))
if not csv_files:
    raise FileNotFoundError(f"No files match: {pattern}")

topdown_files = []  # for optional collage

for csv_path in csv_files:
    seq_name = os.path.splitext(os.path.basename(csv_path))[0]  # e.g., ctin_mh01
    print(f"[RUN] {seq_name}")

    # --- Load CSV ---
    df = pd.read_csv(csv_path).dropna()
    for c in ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','vx','vy','vz','timestamp']:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c} in {seq_name}")

    has_gt_pos = set(['gt_x','gt_y','gt_z']).issubset(df.columns)

    imu    = df[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']].to_numpy(dtype=np.float32)
    vel_gt = df[['vx','vy','vz']].to_numpy(dtype=np.float64)
    t_all  = df['timestamp'].to_numpy(dtype=np.float64)

    # --- Per-file normalization ---
    imu = np.clip(imu, -IMU_CLIP, IMU_CLIP)
    m = imu.mean(axis=0, dtype=np.float64)
    s = np.maximum(imu.std(axis=0, dtype=np.float64), STD_FLOOR)
    imu = (imu - m.astype(np.float32)) / s.astype(np.float32)
    imu = np.clip(imu, -Z_CLIP, Z_CLIP)

    # --- Windows (center cadence) ---
    X_windows = create_windows(imu, window_size, stride)
    if X_windows.shape[0] == 0:
        print(f"[SKIP] no windows for {seq_name}")
        continue

    Nw  = X_windows.shape[0]
    mid = window_size // 2
    center_idx = np.arange(mid, mid + stride*Nw, stride)
    t_center   = t_all[center_idx]

    # --- Inference ---
    vel_list = []
    with torch.no_grad():
        X = torch.tensor(X_windows, dtype=torch.float32, device=device)
        X = torch.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
        for i in range(0, X.shape[0], batch_size):
            pv, _ = model(X[i:i+batch_size])              # pv: [B,T,3]
            vel_list.append(pv[:, mid, :].detach().cpu())

    vel_pred = torch.cat(vel_list, dim=0).numpy().astype(np.float64)

    # Optional de-bias (helps drift in plots)
    vel_pred = vel_pred - vel_pred.mean(axis=0, keepdims=True)

    # --- Positions (center cadence) ---
    pos_pred = trapezoid_integrate(vel_pred, t_center)  # [N,3]

    if has_gt_pos:
        pos_gt_all = df[['gt_x','gt_y','gt_z']].to_numpy(dtype=np.float64)
        pos_gt = pos_gt_all[center_idx] - pos_gt_all[center_idx[0]]
    else:
        v_gt_c = vel_gt[center_idx]
        pos_gt = trapezoid_integrate(v_gt_c, t_center)

    # --- Align & save top-down plot ---
    PR_aligned, _, _ = align_se3(pos_pred.copy(), pos_gt.copy())
    png, svg = save_topdown_plot(seq_name, pos_gt, PR_aligned)
    topdown_files.append((seq_name, png))

print(f"\n[OK] Saved per-sequence top-down plots to: {OUT_DIR}")

# ========= (Optional) Collage with all sequences =========
if topdown_files:
    # load the PNGs and tile them into a grid figure
    cols = 3
    rows = int(np.ceil(len(topdown_files) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6*cols, 4.6*rows))
    axes = np.atleast_2d(axes)
    for idx, (name, png) in enumerate(topdown_files):
        r, c = divmod(idx, cols)
        img = plt.imread(png)
        axes[r, c].imshow(img)
        axes[r, c].axis('off')
        axes[r, c].set_title(name, fontsize=10, pad=6)
    # hide unused axes
    for k in range(len(topdown_files), rows*cols):
        r, c = divmod(k, cols)
        axes[r, c].axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ALL_topdown_collage.png"), dpi=200)
    plt.close(fig)
    print(f"[OK] Collage saved: {os.path.join(OUT_DIR, 'ALL_topdown_collage.png')}")
