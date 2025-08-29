# tlio_train_NLL.py
import os, time, math, random, platform
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ctin_project.model.ctin_model import CTINModel
from ctin_project.loss.NLL_loss import ctin_loss
from ctin_project.sequence_window_dataset import SequenceWindowDataset

# ---------------- Config ----------------
train_csv_dir   = "../ctin_csv_output/train"
val_csv_dir     = "../ctin_csv_output/validation"
window_size     = 200
stride          = 10
batch_size      = 64
num_epochs      = 25
learning_rate   = 1e-4
save_path       = "../ctin_model_tlio_GRU_NLL.pth"

USE_HUBER_FOR_VEL = False
LAM_VEL           = 1.0
LAM_NLL_VEL       = 1.0
LAM_SMOOTH_VEL    = 1e-3
LOGSTD_REG_LAMBDA = 1e-5
WARMUP_EPOCHS_COV = 5
GRAD_MAX_NORM     = 1.0

SEED = 42

def run_epoch(model, loader, optimizer, device, train_mode: bool, epoch_idx: int):
    model.train(mode=train_mode)
    total_sum = mse_sum = nll_sum = 0.0
    nbatches = 0
    last_stats = {}

    for X, Y in loader:
        X = X.to(device); Y = Y.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            pred_vel, pred_logstd = model(X)  # [B,T,D] each

            assert pred_vel.shape == Y.shape, f"pred_vel {pred_vel.shape} vs Y {Y.shape}"
            assert pred_logstd.shape == Y.shape, f"pred_logstd {pred_logstd.shape} vs Y {Y.shape}"

            pred_vel    = torch.nan_to_num(pred_vel,    nan=0.0, posinf=1e3, neginf=-1e3)
            pred_logstd = torch.nan_to_num(pred_logstd, nan=0.0)

            loss, logs = ctin_loss(
                pred_vel=pred_vel, target_vel=Y, logstd_vel=pred_logstd,
                pred_pos=None, target_pos=None, logstd_pos=None,
                epoch=epoch_idx,
                use_mse_for_vel=(not USE_HUBER_FOR_VEL),
                lam_vel=LAM_VEL,
                lam_nll_vel=LAM_NLL_VEL,
                lam_smooth_vel=LAM_SMOOTH_VEL,
                lambda_logstd_reg=LOGSTD_REG_LAMBDA,
                warmup_epochs_cov=WARMUP_EPOCHS_COV,
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

        # quick z-coverage on last batch
        with torch.no_grad():
            sigma = pred_logstd.exp()
            z = (Y - pred_vel) / (sigma + 1e-8)
            z_abs = z.abs()
            last_stats = {
                "z_cov1": (z_abs <= 1.0).float().mean().item(),
                "z_cov2": (z_abs <= 2.0).float().mean().item(),
                "z_cov3": (z_abs <= 3.0).float().mean().item(),
                "logsig_mean": pred_logstd.mean().item(),
                "logsig_p05":  pred_logstd.quantile(0.05).item(),
                "logsig_p95":  pred_logstd.quantile(0.95).item(),
            }

    avg_total = total_sum / max(1, nbatches)
    avg_mse   = mse_sum   / max(1, nbatches)
    avg_nll   = nll_sum   / max(1, nbatches)
    return avg_total, avg_mse, avg_nll, last_stats

def main():
    # seeds
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # choose workers safely
    NUM_WORKERS = 0 if platform.system() == "Windows" else 4
    PIN_MEMORY  = platform.system() != "Windows"

    # Data
    train_set = SequenceWindowDataset(train_csv_dir, window_size=window_size, stride=stride)
    val_set   = SequenceWindowDataset(val_csv_dir,   window_size=window_size, stride=stride)
    print(f"TLIO_golden windows — Train: {len(train_set)} | Val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, persistent_workers=False)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, persistent_workers=False)

    # Device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = CTINModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    start_time = time.time()
    best_val = math.inf
    for epoch in range(num_epochs):
        train_total, train_mse, train_nll, _ = run_epoch(model, train_loader, optimizer, device, True,  epoch)
        with torch.no_grad():
            val_total, val_mse, val_nll, stats = run_epoch(model, val_loader, optimizer, device, False, epoch)

        elapsed_min = (time.time() - start_time) / 60.0
        print(
            f"Epoch {epoch+1:03d} | "
            f"Train: Total {train_total:.5f} | VelMSE {train_mse:.5f} | VelNLL {train_nll:.5f}  ||  "
            f"Val: Total {val_total:.5f} | VelMSE {val_mse:.5f} | VelNLL {val_nll:.5f}  ||  "
            f"Val z-cov: ≤1σ {stats['z_cov1']*100:5.1f}%  ≤2σ {stats['z_cov2']*100:5.1f}%  ≤3σ {stats['z_cov3']*100:5.1f}%  ||  "
            f"logσμ {stats['logsig_mean']:.3f} [p05 {stats['logsig_p05']:.3f}, p95 {stats['logsig_p95']:.3f}]  ||  "
            f"t+ {elapsed_min:.1f} min"
        )

        if val_total < best_val:
            best_val = val_total
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ↳ Saved best model → {save_path} (best val {best_val:.5f})")

    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to: {save_path}")

if __name__ == "__main__":
    # On Windows, this guard is REQUIRED when using DataLoader workers
    main()
