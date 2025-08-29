# losses_ctin.py
import math
import torch
import torch.nn.functional as F

# --- stability bounds for log-std ---
MIN_LOG_STD = math.log(1e-3)   # σ >= 1e-3
MAX_LOG_STD = math.log(1e+1)   # σ <= 10

def _clamp_logstd(log_std):
    return torch.clamp(log_std, MIN_LOG_STD, MAX_LOG_STD)

def mse_loss(pred, target, reduction: str = "mean"):
    return F.mse_loss(pred, target, reduction=reduction)

def huber_loss(pred, target, delta=0.5, reduction="mean"):
    return F.huber_loss(pred, target, delta=delta, reduction=reduction)

def gaussian_nll_diag(residual, log_std, reduction: str = "mean"):
    """
    residual, log_std: [..., D]
    NLL = 0.5 * (r^2 / σ^2 + 2 log σ). (const term dropped)
    """
    log_std = _clamp_logstd(log_std)
    inv_var = torch.exp(-2.0 * log_std)            # 1/σ^2
    nll = 0.5 * (residual**2) * inv_var + log_std  # per-element (per-dim)
    # reduce over last dim first to treat D dims equally
    nll = nll.mean(dim=-1)
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    else:
        return nll

def temporal_smoothness(log_std, time_dim=-2, reduction="mean"):
    """
    L1 smoothness on log-σ over time axis.
    log_std: [B, T, D] (or [B, W, D] for position windows)
    """
    diffs = log_std.narrow(time_dim, 1, log_std.size(time_dim)-1) - \
            log_std.narrow(time_dim, 0, log_std.size(time_dim)-1)
    diffs = diffs.abs()
    if reduction == "mean":
        return diffs.mean()
    elif reduction == "sum":
        return diffs.sum()
    else:
        return diffs

def ctin_loss(
    # --- velocity head ---
    pred_vel,             # [B, T, 2]
    target_vel,           # [B, T, 2]
    logstd_vel=None,      # [B, T, 2] or None if no vel σ
    # --- optional position head (after integration over window) ---
    pred_pos=None,        # [B, W, 3]  (if you supervise position)
    target_pos=None,      # [B, W, 3]
    logstd_pos=None,      # [B, W, 3]  (σ for position)
    # --- knobs ---
    epoch: int = 0,
    warmup_epochs_cov: int = 10,
    use_mse_for_vel: bool = True,      # else Huber
    lam_vel: float = 1.0,
    lam_nll_vel: float = 0.0,          # set >0 to train σ on velocity
    lam_nll_pos: float = 0.0,          # set >0 to train σ on position
    lam_smooth_vel: float = 0.0,
    lam_smooth_pos: float = 0.0,
    lambda_logstd_reg: float = 0.0,    # tiny L2 on log-σ
    reduction: str = "mean",
):
    """
    Flexible loss:
      L = lam_vel * L_vel_mean +
          lam_nll_vel * NLL_vel(pred_vel, target_vel, logstd_vel) +
          lam_nll_pos * NLL_pos(pred_pos, target_pos, logstd_pos) +
          lam_smooth_* * TV(logσ_*)
    Notes:
      - For your calibration plot on position, set lam_nll_pos > 0 and provide pred_pos/target_pos/logstd_pos.
      - During warm-up, covariance heads are detached to let the mean stabilize.
    """
    # --- 1) Mean velocity regression ---
    if use_mse_for_vel:
        L_vel_mean = mse_loss(pred_vel, target_vel, reduction=reduction)
    else:
        L_vel_mean = huber_loss(pred_vel, target_vel, delta=0.5, reduction=reduction)

    total = lam_vel * L_vel_mean

    # --- 2) Velocity NLL (if requested) ---
    L_nll_vel = torch.tensor(0.0, device=pred_vel.device)
    L_tv_vel  = torch.tensor(0.0, device=pred_vel.device)

    if lam_nll_vel > 0.0 and logstd_vel is not None:
        logstd_v = logstd_vel
        if epoch < warmup_epochs_cov:
            logstd_v = logstd_v.detach()
        res_v = target_vel - pred_vel
        L_nll_vel = gaussian_nll_diag(res_v, logstd_v, reduction=reduction)
        total += lam_nll_vel * L_nll_vel

        if lam_smooth_vel > 0.0:
            L_tv_vel = temporal_smoothness(_clamp_logstd(logstd_v), time_dim=1, reduction=reduction)
            total += lam_smooth_vel * L_tv_vel

        if lambda_logstd_reg > 0.0:
            total += lambda_logstd_reg * (logstd_v**2).mean()

    # --- 3) Position NLL (if requested) ---
    L_nll_pos = torch.tensor(0.0, device=pred_vel.device)
    L_tv_pos  = torch.tensor(0.0, device=pred_vel.device)

    if lam_nll_pos > 0.0 and (pred_pos is not None) and (target_pos is not None) and (logstd_pos is not None):
        logstd_p = logstd_pos
        if epoch < warmup_epochs_cov:
            logstd_p = logstd_p.detach()
        res_p = target_pos - pred_pos
        L_nll_pos = gaussian_nll_diag(res_p, logstd_p, reduction=reduction)
        total += lam_nll_pos * L_nll_pos

        if lam_smooth_pos > 0.0:
            L_tv_pos = temporal_smoothness(_clamp_logstd(logstd_p), time_dim=1, reduction=reduction)
            total += lam_smooth_pos * L_tv_pos

        if lambda_logstd_reg > 0.0:
            total += lambda_logstd_reg * (logstd_p**2).mean()

    # Return components for logging
    return total, {
        "L_vel_mean": L_vel_mean.detach(),
        "L_nll_vel":  L_nll_vel.detach(),
        "L_tv_vel":   L_tv_vel.detach(),
        "L_nll_pos":  L_nll_pos.detach(),
        "L_tv_pos":   L_tv_pos.detach(),
    }
