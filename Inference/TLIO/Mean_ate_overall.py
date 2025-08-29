# pdr_ctin_summary.py
import os, re
import numpy as np
import pandas as pd
from pathlib import Path
import torch

from ctin_project.model.ctin_model import CTINModel  # must return (vel, logstd) -> [B,T,2] each

# ====== Config ======
CTIN_CSV_ROOT = Path("../../ctin_csv_output")
TRAIN_DIR     = CTIN_CSV_ROOT / "train"
VAL_DIR       = CTIN_CSV_ROOT / "validation"

MODEL_PATH    = Path("../../ctin_model_tlio_GRU_NLL.pth")

WINDOW  = 200
STRIDE  = 20         # match your inference saving
BATCH   = 256
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMU normalization (match training)
STD_FLOOR = 1e-3
IMU_CLIP  = 80.0
Z_CLIP    = 10.0

# Expected columns (rename if yours differ)
IMU_COLS = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
VEL_COLS = ["vel_x","vel_y"]       # 2D velocity (m/s)
POS_COLS = ["gt_x","gt_y"]         # optional (abs. GT pos)
TIME_COL = "timestamp"             # seconds

OUT_DIR = Path("./PDR/summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== Helpers ======
def create_windows(data, window, stride):
    n = len(data); out = []
    for i in range(0, n - window + 1, stride):
        w = data[i:i+window]
        if w.shape[0] == window: out.append(w)
    if not out: return np.zeros((0, window, data.shape[1]), dtype=data.dtype)
    return np.stack(out)

def integrate_trap(vel, t):
    N = vel.shape[0]; pos = np.zeros_like(vel, dtype=np.float64)
    if N <= 1: return pos
    dt = np.diff(t).astype(np.float64)
    dt = np.clip(dt, 1e-4, 1.0)
    incr = 0.5 * (vel[:-1] + vel[1:]) * dt[:, None]
    pos[1:] = np.cumsum(incr, axis=0)
    return pos

def ate_metrics_from_pos(pos_pred, pos_gt):
    err = pos_pred - pos_gt                        # [N,2]
    err_norm = np.linalg.norm(err, axis=1)         # per-step ATE
    metrics = {
        "ate_mean_m": float(err_norm.mean()),
        "ate_p50_m":  float(np.median(err_norm)),
        "ate_p95_m":  float(np.percentile(err_norm, 95)),
        "traj_rmse_m": float(np.sqrt(np.mean(np.sum(err**2, axis=1)))),  # ATE-RMSE
        "rmse_x_m": float(np.sqrt(np.mean(err[:,0]**2))),
        "rmse_y_m": float(np.sqrt(np.mean(err[:,1]**2))),
        "N": int(len(err_norm)),
    }
    return metrics

def run_ctin_on_csv(csv_path: Path, model: CTINModel):
    df = pd.read_csv(csv_path).dropna()
    for c in IMU_COLS + VEL_COLS + [TIME_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {csv_path}")

    has_gt_pos = all(c in df.columns for c in POS_COLS)

    imu     = df[IMU_COLS].to_numpy(dtype=np.float32)
    vel_gtA = df[VEL_COLS].to_numpy(dtype=np.float64)  # [T,2]
    t_all   = df[TIME_COL].to_numpy(dtype=np.float64)

    # Per-file z-norm on IMU
    imu = np.clip(imu, -IMU_CLIP, IMU_CLIP)
    m = imu.mean(axis=0, dtype=np.float64)
    s = np.maximum(imu.std(axis=0, dtype=np.float64), STD_FLOOR)
    imu = (imu - m.astype(np.float32)) / s.astype(np.float32)
    imu = np.clip(imu, -Z_CLIP, Z_CLIP)

    # Windowing on IMU, center-cadence outputs
    Xw = create_windows(imu, WINDOW, STRIDE)
    if Xw.shape[0] == 0:
        raise RuntimeError(f"No windows for {csv_path.name} (len={len(imu)})")

    mid = WINDOW // 2
    Nw  = Xw.shape[0]
    center_idx = np.arange(mid, mid + STRIDE*Nw, STRIDE)
    if center_idx[-1] >= len(df):
        center_idx = center_idx[center_idx < len(df)]
        Xw = Xw[:len(center_idx)]
    t_c    = t_all[center_idx]
    vel_gt = vel_gtA[center_idx]

    # Choose GT position source
    if has_gt_pos:
        pos_gt = df[POS_COLS].to_numpy(dtype=np.float64)[center_idx]
        pos_gt = pos_gt - pos_gt[0]                   # origin shift
    else:
        pos_gt = integrate_trap(vel_gt, t_c)

    # Model forward (center frame)
    vel_list = []
    with torch.no_grad():
        X = torch.tensor(Xw, dtype=torch.float32, device=DEVICE)
        X = torch.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
        for i in range(0, X.shape[0], BATCH):
            pv, _ = model(X[i:i+BATCH])              # pv: [B,W,2]
            vel_list.append(pv[:, mid, :].detach().cpu())
    vel_pred = torch.cat(vel_list, dim=0).numpy().astype(np.float64)

    # Optional slight de-bias of vel means
    vel_pred = vel_pred - vel_pred.mean(axis=0, keepdims=True)

    pos_pred = integrate_trap(vel_pred, t_c)

    # Metrics
    metrics = ate_metrics_from_pos(pos_pred, pos_gt)

    # Add velocity RMSE too (nice to have)
    v_rmse = np.sqrt(np.mean((vel_pred - vel_gt)**2, axis=0))
    metrics["vel_rmse_vx_ms"] = float(v_rmse[0])
    metrics["vel_rmse_vy_ms"] = float(v_rmse[1])

    return metrics

def summarize_split(split_name: str, dir_path: Path, model: CTINModel):
    rows = []
    files = sorted(dir_path.glob("*.csv"))
    if not files:
        print(f"[WARN] No CSVs in {dir_path}")
        return None

    for f in files:
        try:
            m = run_ctin_on_csv(f, model)
            m["seq"] = f.stem
            rows.append(m)
            print(f"[{split_name}] {f.name} -> ATEmean {m['ate_mean_m']:.3f} m | RMSE {m['traj_rmse_m']:.3f} m")
        except Exception as e:
            print(f"[SKIP] {f.name}: {e}")

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("seq")
    df.to_csv(OUT_DIR / f"{split_name}_per_sequence.csv", index=False)

    # Macro-average (unweighted) and sample-weighted average
    macro_ate  = df["ate_mean_m"].mean()
    macro_rmse = df["traj_rmse_m"].mean()

    w = df["N"].to_numpy()
    w = w / w.sum()
    w_ate  = float((df["ate_mean_m"] * w).sum())
    w_rmse = float((df["traj_rmse_m"] * w).sum())

    summary = {
        "split": split_name,
        "num_seqs": int(len(df)),
        "macro_ate_mean_m": float(macro_ate),
        "macro_traj_rmse_m": float(macro_rmse),
        "weighted_ate_mean_m": w_ate,
        "weighted_traj_rmse_m": w_rmse,
    }
    print(f"\n[{split_name} SUMMARY] "
          f"macro ATE {macro_ate:.3f} m | macro RMSE {macro_rmse:.3f} m | "
          f"weighted ATE {w_ate:.3f} m | weighted RMSE {w_rmse:.3f} m\n")
    return df, summary

# ====== Main ======
def main():
    # Load model once
    model = CTINModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_summaries = []

    for split_name, dir_path in [("training", TRAIN_DIR), ("validation", VAL_DIR)]:
        res = summarize_split(split_name, dir_path, model)
        if res is not None:
            _, summary = res
            all_summaries.append(summary)

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(OUT_DIR / "ctin_pdr_split_summary.csv", index=False)
        print(f"[OK] Wrote split summaries -> {OUT_DIR / 'ctin_pdr_split_summary.csv'}")
    else:
        print("[WARN] No summaries produced.")

if __name__ == "__main__":
    main()
