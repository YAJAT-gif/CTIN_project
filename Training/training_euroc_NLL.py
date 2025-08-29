# train_euroc_ctin_val.py
import os, time, math, random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from ctin_project.model.ctin_model import CTINModel   # returns (pred_vel, pred_logstd) with shapes [B,T,3]
from ctin_project.sequence_window_dataset_EUROC import SequenceWindowDataset
from ctin_project.loss.NLL_loss import ctin_loss      # must return (total, logs) where logs has 'L_vel_mean','L_nll_vel'

# ---------------- Config ----------------
csv_dir        = "../euroc_output"
window_size    = 200
stride         = 5
batch_size     = 64
num_epochs     = 100
learning_rate  = 1e-4
save_path      = "../ctin_model_EUROC_GRU_3D_NLL.pth"

VAL_FRACTION        = 0.1      # 10% validation
SHUFFLE_SEED        = 42       # reproducible split
USE_HUBER_FOR_VEL   = False    # True -> Huber, False -> MSE
LAM_VEL             = 1.0
LAM_NLL_VEL         = 1.0
LAM_SMOOTH_VEL      = 1e-3
LOGSTD_REG_LAMBDA   = 1e-5
WARMUP_EPOCHS_COV   = 10
GRAD_MAX_NORM       = 1.0

# ---------------- Determinism (helpful for reproducibility) ----------------
torch.manual_seed(SHUFFLE_SEED)
np.random.seed(SHUFFLE_SEED)
random.seed(SHUFFLE_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- Data ----------------
dataset = SequenceWindowDataset(csv_dir, window_size=window_size, stride=stride)
n_total = len(dataset)
n_val   = max(1, int(VAL_FRACTION * n_total))
n_train = n_total - n_val

train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SHUFFLE_SEED))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=False)

print(f"Dataset windows: {n_total}  |  Train: {n_train}  Val: {n_val}")
print(f"Num train batches/epoch: {len(train_loader)}  |  Num val batches: {len(val_loader)}")

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Model & Optimizer ----------------
model = CTINModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---------------- Helpers ----------------
def run_epoch(loader, train_mode: bool, epoch_idx: int):
    """
    Returns: avg_total, avg_mse, avg_nll, last_batch_stats_dict
    last_batch_stats_dict contains z-coverage and logσ summary for quick inspection.
    """
    if train_mode:
        model.train()
    else:
        model.eval()

    total_sum = 0.0
    mse_sum   = 0.0
    nll_sum   = 0.0
    nbatches  = 0

    last_stats = {}

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            pred_vel, pred_logstd = model(X)  # [B,T,3] each

            pred_vel    = torch.nan_to_num(pred_vel,    nan=0.0, posinf=1e3, neginf=-1e3)
            pred_logstd = torch.nan_to_num(pred_logstd, nan=0.0)

            # Compute loss (MSE/Huber on velocity mean + NLL on velocity σ)
            loss, logs = ctin_loss(
                pred_vel=pred_vel, target_vel=Y, logstd_vel=pred_logstd,
                pred_pos=None, target_pos=None, logstd_pos=None,
                epoch=epoch_idx,
                use_mse_for_vel=not USE_HUBER_FOR_VEL,  # True->MSE, False->Huber in our wrapper
                lam_vel=LAM_VEL,
                lam_nll_vel=LAM_NLL_VEL,
                lam_smooth_vel=LAM_SMOOTH_VEL,
                lambda_logstd_reg=LOGSTD_REG_LAMBDA,
                reduction="mean",
            )

            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_MAX_NORM)
                optimizer.step()

        total_sum += float(loss.item())
        mse_sum   += float(logs["L_vel_mean"].item())
        nll_sum   += float(logs["L_nll_vel"].item())
        nbatches  += 1

        # Quick z-score coverage on this (last) batch
        with torch.no_grad():
            sigma = pred_logstd.exp()
            z = (Y - pred_vel) / (sigma + 1e-8)
            z_abs = z.abs()
            cov1 = (z_abs <= 1.0).float().mean().item()
            cov2 = (z_abs <= 2.0).float().mean().item()
            cov3 = (z_abs <= 3.0).float().mean().item()
            last_stats = {
                "z_cov1": cov1, "z_cov2": cov2, "z_cov3": cov3,
                "logsig_mean": pred_logstd.mean().item(),
                "logsig_p05": pred_logstd.quantile(0.05).item(),
                "logsig_p95": pred_logstd.quantile(0.95).item(),
            }

    avg_total = total_sum / max(1, nbatches)
    avg_mse   = mse_sum   / max(1, nbatches)
    avg_nll   = nll_sum   / max(1, nbatches)
    return avg_total, avg_mse, avg_nll, last_stats

# ---------------- Train Loop ----------------
start_time = time.time()
best_val = math.inf

for epoch in range(num_epochs):
    # Train
    train_total, train_mse, train_nll, train_stats = run_epoch(train_loader, True, epoch)

    # Validate (no grad)
    with torch.no_grad():
        val_total, val_mse, val_nll, val_stats = run_epoch(val_loader, False, epoch)

    elapsed = time.time() - start_time
    print(
        f"Epoch {epoch+1:03d} | "
        f"Train: Total {train_total:.5f} | MSE {train_mse:.5f} | NLL {train_nll:.5f} || "
        f"Val: Total {val_total:.5f} | MSE {val_mse:.5f} | NLL {val_nll:.5f} || "
        f"z_cov (val) ≤1σ {val_stats['z_cov1']*100:5.1f}% ≤2σ {val_stats['z_cov2']*100:5.1f}% ≤3σ {val_stats['z_cov3']*100:5.1f}% || "
        f"logσ_mean {val_stats['logsig_mean']:.3f} [p05 {val_stats['logsig_p05']:.3f}, p95 {val_stats['logsig_p95']:.3f}] || "
        f"t+ {elapsed/60:.1f} min"
    )

    # Save best by validation total loss
    if val_total < best_val:
        best_val = val_total
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  ↳ Saved best model → {save_path} (best val {best_val:.5f})")

# Final save (optional)
torch.save(model.state_dict(), save_path)
print(f"Final model saved to: {save_path}")
