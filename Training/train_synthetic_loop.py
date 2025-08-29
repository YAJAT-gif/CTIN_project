import os, glob, json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from ctin_project.model.ctin_model import CTINModel
from ctin_project.loss.NLL_loss import ctin_loss  # your loss file

# ========= Config =========
ROOT     = "../Datasets"
IMU_DIR  = os.path.join(ROOT, "imu_matlab")
GT_DIR   = os.path.join(ROOT, "gt_matlab")

WINDOW_SIZE = 200
STRIDE      = 1
BATCH_SIZE  = 64
VEL_SCALE   = 40.0

LR          = 3e-4           # safer for NLL
WEIGHT_DECAY= 1e-4
EPOCHS      = 100
SAVE_MODEL  = "ctin_matlab_trainonly_GRU_NLL.pth"
SAVE_NORM   = "ctin_matlab_norm.json"
CKPT_EVERY  = 20              # save every N epochs

# MATLAB IMU order is [gyro, accel]; set True if model expects [accel, gyro]
SWAP_IMU_TO_ACC_FIRST = True

# NLL knobs
LAM_VEL        = 1.0
LAM_NLL_VEL_0  = 0.0         # ramp NLL weight
LAM_NLL_VEL_1  = 0.5
LAM_NLL_VEL_2  = 1.0
WARMUP_COV_E   = 10          # detach σ for first N epochs in loss
LAM_SMOOTH_VEL = 5e-3
LOGSTD_L2      = 1e-5
USE_MSE        = True

# ========= Helpers =========
def load_csv(fp): return pd.read_csv(fp, header=None).values.astype(np.float32)

def find_pairs(imu_dir, gt_dir):
    imu_files = sorted(glob.glob(os.path.join(imu_dir, "imu_ctin_*.csv")))
    pairs = []
    for imu_fp in imu_files:
        base = os.path.basename(imu_fp).replace("imu_ctin_", "").replace(".csv", "")
        gt_fp = os.path.join(gt_dir, f"gt_vel_ctin_{base}.csv")
        if os.path.exists(gt_fp):
            pairs.append((imu_fp, gt_fp, base))
        else:
            print(f"[WARN] Missing GT for {imu_fp} -> {gt_fp}")
    if not pairs:
        raise FileNotFoundError(f"No (imu,gt) pairs in {imu_dir} / {gt_dir}")
    return pairs

def compute_stats(pairs):
    rows = []
    for imu_fp, _, _ in pairs:
        imu = load_csv(imu_fp)
        if SWAP_IMU_TO_ACC_FIRST:
            imu = imu[:, [3,4,5, 0,1,2]]  # [acc, gyro]
        rows.append(imu)
    all_imu = np.concatenate(rows, axis=0)
    mean = all_imu.mean(axis=0).astype(np.float32)
    std  = np.clip(all_imu.std(axis=0).astype(np.float32), 1e-6, None)
    return mean, std

def save_norm(path, imu_mean, imu_std, vel_scale):
    with open(path, "w") as f:
        json.dump({
            "imu_mean": imu_mean.tolist(),
            "imu_std":  imu_std.tolist(),
            "vel_scale": vel_scale,
            "swap_acc_first": SWAP_IMU_TO_ACC_FIRST
        }, f, indent=2)

# ========= Dataset =========
class MatlabPairsWindowed(Dataset):
    def __init__(self, pairs, imu_mean, imu_std, window_size=200, stride=1, vel_scale=40.0):
        self.samples = []
        self.W = window_size; self.S = stride
        self.imu_mean = imu_mean.astype(np.float32)
        self.imu_std  = imu_std.astype(np.float32)
        self.vel_scale= vel_scale

        for imu_fp, gt_fp, _ in pairs:
            imu = load_csv(imu_fp)                 # [T,6] (gyro,accel)
            if SWAP_IMU_TO_ACC_FIRST:
                imu = imu[:, [3,4,5, 0,1,2]]       # -> [acc, gyro]
            imu = (imu - self.imu_mean) / self.imu_std

            vel = load_csv(gt_fp) / self.vel_scale # [T,2]

            T = min(len(imu), len(vel))
            imu, vel = imu[:T], vel[:T]

            for i in range(0, T - self.W + 1, self.S):
                Xi = imu[i:i+self.W, :]
                Yi = vel[i:i+self.W, :]
                self.samples.append((Xi, Yi))

        if not self.samples:
            raise RuntimeError("No windows created — check window_size/stride vs file lengths.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        Xi, Yi = self.samples[idx]
        return torch.from_numpy(Xi), torch.from_numpy(Yi)

# ========= Build train-only loader =========
pairs = find_pairs(IMU_DIR, GT_DIR)
imu_mean, imu_std = compute_stats(pairs)
save_norm(SAVE_NORM, imu_mean, imu_std, VEL_SCALE)
print(f"[INFO] Saved normalization -> {SAVE_NORM}")

train_ds = MatlabPairsWindowed(pairs, imu_mean, imu_std, WINDOW_SIZE, STRIDE, VEL_SCALE)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
print(f"[INFO] Train windows: {len(train_ds)}")

# ========= Model / Optimizer =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTINModel().to(device)                       # must output (pred_vel, logstd_vel)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def sanitize(x, clip_abs=None):
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    if clip_abs is not None:
        x = torch.clamp(x, -clip_abs, clip_abs)
    return x

def nll_weight_for_epoch(e):
    if e <= 5:   return LAM_NLL_VEL_0
    if e <= 10:  return LAM_NLL_VEL_1
    return LAM_NLL_VEL_2

def run_epoch(loader, epoch):
    model.train()
    tot = vel_m = nll_v = tv_v = 0.0; n = 0
    for Xb, Yb in loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        Xb = sanitize(Xb, clip_abs=20.0)

        optimizer.zero_grad()
        pred_vel, logstd_vel = model(Xb)
        pred_vel   = sanitize(pred_vel,   clip_abs=50.0)
        logstd_vel = sanitize(logstd_vel, clip_abs=6.9)

        lam_nll = nll_weight_for_epoch(epoch)

        loss, terms = ctin_loss(
            pred_vel=pred_vel, target_vel=Yb, logstd_vel=logstd_vel,
            pred_pos=None, target_pos=None, logstd_pos=None,
            epoch=epoch-1, warmup_epochs_cov=WARMUP_COV_E,
            use_mse_for_vel=USE_MSE,
            lam_vel=LAM_VEL, lam_nll_vel=lam_nll, lam_nll_pos=0.0,
            lam_smooth_vel=LAM_SMOOTH_VEL, lam_smooth_pos=0.0,
            lambda_logstd_reg=LOGSTD_L2, reduction="mean"
        )

        if torch.isnan(loss):
            print("[WARN] NaN loss — batch skipped"); continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        tot  += float(loss.detach().cpu())
        vel_m+= float(terms["L_vel_mean"].detach().cpu())
        nll_v+= float(terms["L_nll_vel"].detach().cpu())
        tv_v += float(terms["L_tv_vel"].detach().cpu())
        n += 1

    if n == 0: return float("inf"), float("inf"), float("inf"), float("inf")
    return tot/n, vel_m/n, nll_v/n, tv_v/n

# ========= Train =========
best = float("inf")
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    tr_tot, tr_mse, tr_nllv, tr_tv = run_epoch(train_loader, epoch)
    dt = time.time() - t0

    print(f"Epoch {epoch:02d} | "
          f"Train: Tot {tr_tot:.4f}  VelMSE {tr_mse:.4f}  NLLv {tr_nllv:.4f}  TVσ {tr_tv:.4f} | "
          f"{dt:.1f}s")

    # save best on train (since no val)
    if tr_tot < best:
        best = tr_tot
        torch.save(model.state_dict(), SAVE_MODEL)
        print(f"  -> saved best (train loss {best:.4f}) to {SAVE_MODEL}")

    if (epoch % CKPT_EVERY) == 0:
        ckpt = f"{os.path.splitext(SAVE_MODEL)[0]}_e{epoch}.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"  -> checkpoint: {ckpt}")

print(f"[DONE] Best train total loss: {best:.4f}")
