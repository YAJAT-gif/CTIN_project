# pdr_batch_ctin_vs_tlio.py
import os, re, glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import FancyArrowPatch

# ====== project model (must output (vel, logstd)) ======
from ctin_project.model.ctin_model import CTINModel

# ---------------- Paths & Config ----------------
CTIN_CSV_ROOT   = Path("../../ctin_csv_output")
TRAIN_DIR       = CTIN_CSV_ROOT / "train"
VAL_DIR         = CTIN_CSV_ROOT / "validation"
TLIO_RESULTS    = Path("../../results")                # expects <seqid>/trajectory.txt
NPZ_BASE        = Path("../Inference/TLIO")         # will save <seqid>/ctin_results.npz

MODEL_PATH      = Path("../../ctin_model_tlio_GRU_NLL.pth")
WINDOW          = 200
STRIDE          = 20
BATCH           = 256

# IMU normalization (match training)
STD_FLOOR = 1e-3; IMU_CLIP = 80.0; Z_CLIP = 10.0

IMU_COLS = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
VEL_COLS = ["vel_x","vel_y"]
POS_COLS = ["gt_x","gt_y"]       # optional
TIME_COL = "timestamp"

# Colors (Okabe–Ito)
COL_GT   = "#000000"
COL_CTIN = "#E69F00"
COL_TLIO = "#009E73"
COL_SCAT = "#0072B2"
COL_CONE = "#D55E00"

OUT_DIR = Path("./PDR/batch_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "savefig.dpi": 300
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Helpers ----------------
def pick_two(dir_path: Path):
    files = sorted(dir_path.glob("*.csv"))
    return files[:2]

def seq_id_from_csv(path: Path):
    # expects ..._###########.csv ; grab trailing digits
    m = re.search(r"(\d+)\.csv$", path.name)
    if not m:
        raise ValueError(f"Cannot parse sequence id from {path.name}")
    return m.group(1)

def create_windows(data, window, stride):
    n = len(data); out = []
    for i in range(0, n - window + 1, stride):
        w = data[i:i+window]
        if w.shape[0] == window: out.append(w)
    if not out: return np.zeros((0, window, data.shape[1]), dtype=data.dtype)
    return np.stack(out)

def trapezoid_integrate(vel, t):
    N = vel.shape[0]; pos = np.zeros_like(vel, dtype=np.float64)
    if N <= 1: return pos
    dt = np.diff(t).astype(np.float64); dt = np.clip(dt, 1e-4, 1.0)
    incr = 0.5 * (vel[:-1] + vel[1:]) * dt[:, None]
    pos[1:] = np.cumsum(incr, axis=0); return pos

def interp_traj(t_src, P_src, t_tgt):
    out = np.empty((len(t_tgt), P_src.shape[1]), dtype=np.float64)
    for d in range(P_src.shape[1]):
        out[:, d] = np.interp(t_tgt, t_src, P_src[:, d])
    return out

def metrics_xy(P, GT):
    e = P - GT
    e_norm = np.linalg.norm(e, axis=1)
    return dict(
        mean=float(e_norm.mean()),
        p95=float(np.percentile(e_norm, 95)),
        rmse=float(np.sqrt(np.mean((P-GT)**2))),
        max=float(e_norm.max()),
        series=e_norm
    )

def arrow_at_end(ax, P, color):
    if len(P) < 2: return
    a = FancyArrowPatch((P[-2,0], P[-2,1]), (P[-1,0], P[-1,1]),
                        arrowstyle="-|>", mutation_scale=10, lw=0, color=color, zorder=6)
    ax.add_patch(a)

def coverage_stats(err_axis, sigma_axis):
    z = np.abs(err_axis) / np.clip(sigma_axis, 1e-12, None)
    return (float(np.mean(z <= 1.0)),
            float(np.mean(z <= 2.0)),
            float(np.mean(z <= 3.0)))

# ---------------- Inference (CTIN) -> NPZ ----------------
def run_ctin_and_save_npz(csv_path: Path, npz_path: Path):
    df = pd.read_csv(csv_path).dropna()
    for c in IMU_COLS + VEL_COLS + [TIME_COL]:
        if c not in df.columns: raise ValueError(f"Missing column: {c} in {csv_path.name}")
    has_gt_pos = all(c in df.columns for c in POS_COLS)

    imu = df[IMU_COLS].to_numpy(dtype=np.float32)
    vel_gt_all = df[VEL_COLS].to_numpy(dtype=np.float64)
    t_all = df[TIME_COL].to_numpy(dtype=np.float64)

    # Per-file z-norm
    imu = np.clip(imu, -IMU_CLIP, IMU_CLIP)
    m = imu.mean(axis=0, dtype=np.float64)
    s = np.maximum(imu.std(axis=0, dtype=np.float64), STD_FLOOR)
    imu = (imu - m.astype(np.float32)) / s.astype(np.float32)
    imu = np.clip(imu, -Z_CLIP, Z_CLIP)

    Xw = create_windows(imu, WINDOW, STRIDE)
    if Xw.shape[0] == 0: raise RuntimeError("No windows; adjust window/stride.")
    mid = WINDOW // 2; Nw = Xw.shape[0]
    center_idx = np.arange(mid, mid + STRIDE*Nw, STRIDE)
    if center_idx[-1] >= len(df):
        center_idx = center_idx[center_idx < len(df)]
        Xw = Xw[:len(center_idx)]
    t_center = t_all[center_idx]
    vel_gt   = vel_gt_all[center_idx]
    if has_gt_pos:
        pos_gt = df[POS_COLS].to_numpy(dtype=np.float64)[center_idx]
        pos_gt = pos_gt - pos_gt[0]
    else:
        pos_gt = trapezoid_integrate(vel_gt, t_center)

    # CTIN model
    model = CTINModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    vel_list, logstd_list = [], []
    with torch.no_grad():
        X = torch.tensor(Xw, dtype=torch.float32, device=device)
        X = torch.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
        for i in range(0, X.shape[0], BATCH):
            pv, pl = model(X[i:i+BATCH])  # [B,W,2]
            vel_list.append(pv[:, mid, :].detach().cpu())
            logstd_list.append(pl[:, mid, :].detach().cpu())

    vel_pred_center = torch.cat(vel_list, dim=0).numpy().astype(np.float64)
    logstd_vel      = torch.cat(logstd_list, dim=0).numpy().astype(np.float64)
    sigma_v         = np.exp(logstd_vel).astype(np.float64)

    # small de-bias on velocity mean
    vel_pred_center = vel_pred_center - vel_pred_center.mean(axis=0, keepdims=True)

    pos_pred = trapezoid_integrate(vel_pred_center, t_center)
    pos_gt0  = pos_gt - pos_gt[0]
    pos_pr0  = pos_pred - pos_pred[0]

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        t=t_center.astype(np.float64),
        pos_gt=pos_gt0.astype(np.float64),
        pos_pred=pos_pr0.astype(np.float64),
        vel_gt=vel_gt.astype(np.float64),
        vel_pred=vel_pred_center.astype(np.float64),
        logstd_vel=logstd_vel.astype(np.float64),
        sigma_v=sigma_v.astype(np.float64),
    )
    return npz_path

# ---------------- Plotters ----------------
def save_panel_and_plots(seqid: str, tlio_dir: Path, npz_path: Path):
    # TLIO
    traj_file = tlio_dir / "trajectory.txt"
    if not traj_file.exists():
        raise FileNotFoundError(f"Missing TLIO file: {traj_file}")
    try:
        arr = np.loadtxt(traj_file, delimiter=",")
    except ValueError:
        arr = np.loadtxt(traj_file, delimiter=",", skiprows=1)
    t_tlio = arr[:,0].astype(np.float64)
    P_tlio = arr[:,1:3].astype(np.float64)
    GT_tlio= arr[:,4:6].astype(np.float64)

    # CTIN NPZ
    Z = np.load(npz_path)
    t_ctin  = Z["t"].astype(np.float64)
    GT_ctin = Z["pos_gt"][:, :2].astype(np.float64)
    P_ctin  = Z["pos_pred"][:, :2].astype(np.float64)
    vel_gt  = Z["vel_gt"].astype(np.float64)
    vel_pr  = Z["vel_pred"].astype(np.float64)
    sigma_v = Z["sigma_v"].astype(np.float64)

    # Interp TLIO to CTIN time base (relative)
    t_tlio_rel = t_tlio - t_tlio[0]
    t_ctin_rel = t_ctin - t_ctin[0]
    P_tlio_i   = interp_traj(t_tlio_rel, P_tlio,  t_ctin_rel)
    GT_tlio_i  = interp_traj(t_tlio_rel, GT_tlio, t_ctin_rel)

    # Metrics
    m_ctin = metrics_xy(P_ctin,  GT_ctin)
    m_tlio = metrics_xy(P_tlio_i, GT_tlio_i)

    # Visual alignment by translation to CTIN GT origin
    O   = GT_ctin[0]
    GT0 = GT_ctin - O
    C0  = P_ctin  - O
    T0  = P_tlio_i- O

    # === Panel 2x2 ===
    fig = plt.figure(figsize=(7.2, 5.4))
    fig.suptitle(f"RONIN PDR — {seqid}", y=0.98, fontsize=14, fontweight="bold")

    # (1) CDF
    ax1 = fig.add_subplot(2,2,1)
    s_ctin, s_tlio = np.sort(m_ctin['series']), np.sort(m_tlio['series'])
    y_ctin = np.linspace(0, 1, len(s_ctin))
    y_tlio = np.linspace(0, 1, len(s_tlio))
    ax1.plot(s_ctin, y_ctin, color=COL_CTIN, lw=2, label="CTIN")
    ax1.plot(s_tlio, y_tlio, color=COL_TLIO, lw=2, ls="--", label="TLIO")
    med = float(np.percentile(m_ctin['series'], 50))
    ax1.axvline(med, color=COL_CTIN, ls=":", lw=1)
    ax1.axvline(m_ctin['p95'], color=COL_CTIN, ls="--", lw=1, alpha=0.9)
    ax1.set_xlabel("ATE [m]"); ax1.set_ylabel("CDF")
    ax1.set_title("CDF of ATE"); ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False, loc="lower right")

    # (2) Error vs Time
    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(t_ctin_rel, m_ctin['series'], color=COL_CTIN, lw=2, label="CTIN")
    ax2.plot(t_ctin_rel, m_tlio['series'], color=COL_TLIO, lw=2, ls="--", label="TLIO")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Error [m]")
    ax2.set_title("Error vs Time"); ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False, loc="upper right")

    # (3) Grouped metrics
    ax3 = fig.add_subplot(2,2,3)
    labels = ["ATE mean", "ATE p95", "RMSE", "Max"]
    v_ctin = [m_ctin['mean'], m_ctin['p95'], m_ctin['rmse'], m_ctin['max']]
    v_tlio = [m_tlio['mean'], m_tlio['p95'], m_tlio['rmse'], m_tlio['max']]
    x = np.arange(len(labels)); w = 0.38
    b1 = ax3.bar(x - w/2, v_ctin, width=w, color=COL_CTIN, alpha=0.95, label="CTIN")
    b2 = ax3.bar(x + w/2, v_tlio, width=w, color=COL_TLIO, alpha=0.95, label="TLIO")
    y_max = max(max(v_ctin), max(v_tlio))
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax3.text(b.get_x()+b.get_width()/2, v + 0.02*y_max, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8)
    ax3.set_xticks(x); ax3.set_xticklabels(labels)
    ax3.set_ylabel("Error [m]")
    ax3.set_title("CTIN vs TLIO Metrics")
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.legend(frameon=False, loc="upper left")

    # (4) XY overlay
    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(GT0[:,0], GT0[:,1], color=COL_GT,   lw=2.2, label="GT")
    ax4.plot(C0[:,0],  C0[:,1],  color=COL_CTIN, lw=2.2, ls="--", label="CTIN")
    ax4.plot(T0[:,0],  T0[:,1],  color=COL_TLIO, lw=2.2, ls=":",  label="TLIO")
    for P, col in [(GT0,COL_GT), (C0,COL_CTIN), (T0,COL_TLIO)]:
        ax4.scatter(P[0,0], P[0,1], c=col, s=26, marker="o", zorder=5)
        ax4.scatter(P[-1,0],P[-1,1],c=col, s=34, marker="X", zorder=5)
    arrow_at_end(ax4, C0, COL_CTIN); arrow_at_end(ax4, T0, COL_TLIO)
    ax4.set_aspect("equal", adjustable="box"); ax4.margins(0.03)
    ax4.grid(True, alpha=0.25, linewidth=0.8)
    ax4.set_xlabel("X [m]"); ax4.set_ylabel("Y [m]")
    ax4.set_title("Trajectory Overlay — GT vs CTIN vs TLIO")
    ax4.legend(frameon=False, loc="lower left")

    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(OUT_DIR / f"ronin_panel_{seqid}.png", dpi=300)
    fig.savefig(OUT_DIR / f"ronin_panel_{seqid}.svg")
    plt.close(fig)

    # --- 3σ velocity cone for CTIN ---
    vel_err = vel_pr - vel_gt
    vel_std = Z["sigma_v"]  # already exp(logstd)
    AX_LABS = ["Vx","Vy"]
    fig, axs = plt.subplots(2,1, figsize=(4.2,6.8))
    fig.suptitle(f"{seqid} — Velocity 3σ Cone (CTIN)", y=0.995, fontsize=12, fontweight="bold")
    for i, ax in enumerate(axs):
        e = vel_err[:,i]; s = vel_std[:,i]
        xmax = np.percentile(np.abs(e), 99.5) if e.size else 1.0
        xs = np.linspace(-xmax, xmax, 400)
        ax.plot(xs, np.abs(xs)/3.0, ls="--", lw=1.2, color=COL_CONE)
        ax.fill_between(xs, 0.0, np.abs(xs)/3.0, alpha=0.10, color=COL_CONE)
        ax.scatter(e, s, s=6, alpha=0.35, color=COL_SCAT)
        c1,c2,c3 = coverage_stats(e, s)
        ax.set_title(f"{AX_LABS[i]}  •  ≤3σ: {c3*100:.1f}%")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-xmax, xmax); ax.set_ylim(bottom=0)
        if i==1: ax.set_xlabel("Error (m/s)")
        ax.set_ylabel(r"$\sigma$ (m/s)")
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(OUT_DIR / f"{seqid}_vel_cone.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Save individual smaller plots (Word-friendly) ---
    # Error vs time
    plt.figure(figsize=(5.2,3.2))
    plt.plot(t_ctin_rel, m_ctin['series'], color=COL_CTIN, lw=2, label="CTIN")
    plt.plot(t_ctin_rel, m_tlio['series'], color=COL_TLIO, lw=2, ls="--", label="TLIO")
    plt.xlabel("Time [s]"); plt.ylabel("Error [m]"); plt.title(f"{seqid} — Error vs Time")
    plt.grid(True, alpha=0.3); plt.legend(frameon=False, loc="upper right")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"{seqid}_err_vs_time.png", dpi=300); plt.close()

    # CDF
    s_ctin, s_tlio = np.sort(m_ctin['series']), np.sort(m_tlio['series'])
    y_ctin = np.linspace(0,1,len(s_ctin)); y_tlio = np.linspace(0,1,len(s_tlio))
    plt.figure(figsize=(4.4,3.4))
    plt.plot(s_ctin, y_ctin, color=COL_CTIN, lw=2, label="CTIN")
    plt.plot(s_tlio, y_tlio, color=COL_TLIO, lw=2, ls="--", label="TLIO")
    plt.xlabel("ATE [m]"); plt.ylabel("CDF"); plt.title(f"{seqid} — CDF of ATE")
    plt.grid(True, alpha=0.3); plt.legend(frameon=False, loc="lower right")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"{seqid}_cdf_ate.png", dpi=300); plt.close()

    # Top-down (just overlay)
    fig, ax = plt.subplots(figsize=(4.8,4.8))
    ax.plot(GT0[:,0], GT0[:,1], color=COL_GT,   lw=2.2, label="GT")
    ax.plot(C0[:,0],  C0[:,1],  color=COL_CTIN, lw=2.2, ls="--", label="CTIN")
    ax.plot(T0[:,0],  T0[:,1],  color=COL_TLIO, lw=2.2, ls=":",  label="TLIO")
    for P, col in [(GT0,COL_GT), (C0,COL_CTIN), (T0,COL_TLIO)]:
        ax.scatter(P[0,0], P[0,1], c=col, s=26, marker="o", zorder=5)
        ax.scatter(P[-1,0],P[-1,1],c=col, s=34, marker="X", zorder=5)
    arrow_at_end(ax, C0, COL_CTIN); arrow_at_end(ax, T0, COL_TLIO)
    ax.set_aspect("equal", "box"); ax.margins(0.03); ax.grid(True, alpha=0.25)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_title(f"{seqid} — XY Overlay")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT_DIR / f"{seqid}_overlay.png", dpi=300); plt.close(fig)

    return dict(
        seqid=seqid,
        ctin_mean=m_ctin['mean'], ctin_p95=m_ctin['p95'], ctin_rmse=m_ctin['rmse'], ctin_max=m_ctin['max'],
        tlio_mean=m_tlio['mean'], tlio_p95=m_tlio['p95'], tlio_rmse=m_tlio['rmse'], tlio_max=m_tlio['max'],
    )

# ---------------- Main ----------------
def main():
    train_csvs = pick_two(TRAIN_DIR)
    val_csvs   = pick_two(VAL_DIR)
    chosen = train_csvs + val_csvs
    if not chosen:
        raise FileNotFoundError("No training/validation CSVs found.")

    print("[FILES]")
    for p in chosen: print("  ", p)

    rows = []
    for csv_path in chosen:
        sid = seq_id_from_csv(csv_path)
        tlio_dir = TLIO_RESULTS / sid
        npz_path = NPZ_BASE / sid / "ctin_results.npz"
        # run CTIN -> NPZ (always recompute; change if you want to skip when exists)
        run_ctin_and_save_npz(csv_path, npz_path)
        # make all plots and collect metrics
        rows.append(save_panel_and_plots(sid, tlio_dir, npz_path))

    # summary CSV
    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    print(f"\n[OK] Wrote summary: {OUT_DIR / 'summary_metrics.csv'}")
    print(f"[OK] Plots in: {OUT_DIR}")

if __name__ == "__main__":
    main()
