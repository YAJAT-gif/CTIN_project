import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, learn_uncertainty=True, min_variance=1e-2, reg_weight=0.01):
        super(MultiTaskLoss, self).__init__()
        self.min_variance = min_variance
        self.reg_weight = reg_weight

        if learn_uncertainty:
            self.log_delta_v = nn.Parameter(torch.tensor(0.0))  # log(σ_v)
            self.log_delta_c = nn.Parameter(torch.tensor(0.0))  # log(σ_c)
        else:
            self.register_buffer("log_delta_v", torch.tensor(0.0))
            self.register_buffer("log_delta_c", torch.tensor(0.0))

    def forward(self, vel_pred, cov_pred, vel_gt, pos_gt=None, dt=0.01, return_individual=False):
        B, T, _ = vel_pred.shape

        # === Step 1: Dummy position (if not given) ===
        if pos_gt is None:
            pos_gt = torch.zeros_like(vel_gt).cumsum(dim=1) * dt  # [B, T, 2]

        # === Step 2: Velocity Integration ===
        pos_pred = torch.zeros_like(pos_gt)
        pos_pred[:, 0, :] = vel_pred[:, 0, :] * dt
        for t in range(1, T):
            pos_pred[:, t] = pos_pred[:, t - 1] + vel_pred[:, t - 1] * dt

        # === Step 3: Velocity Loss (Positional + Direct Vel Error) ===
        lv_p = F.mse_loss(pos_pred, pos_gt)
        lv_e = F.mse_loss(vel_pred, vel_gt)
        lv_total = lv_p + lv_e

        # === Step 4: Covariance Loss (NLL from 2D Gaussian) ===
        sigma_xx = torch.clamp(torch.exp(cov_pred[..., 0]), min=self.min_variance)
        sigma_yy = torch.clamp(torch.exp(cov_pred[..., 1]), min=self.min_variance)

        # Assume zero off-diagonal (σ_xy = 0) due to Cholesky or diagonal assumption
        det = sigma_xx * sigma_yy
        inv_xx = 1.0 / sigma_xx
        inv_yy = 1.0 / sigma_yy

        diff = vel_gt - vel_pred
        dx = diff[..., 0]
        dy = diff[..., 1]


        mahal = inv_xx * dx**2 + inv_yy * dy**2
        log_det = torch.log(det + 1e-6)
        lc_total = 0.5 * (mahal + log_det).mean()

        # === Step 5: Combine Multi-task Loss ===
        delta_v = torch.exp(self.log_delta_v)
        delta_c = torch.exp(self.log_delta_c)

        total_loss = (
            (lv_total / (2 * delta_v**2)) +
            (lc_total / (2 * delta_c**2)) +
            self.log_delta_v + self.log_delta_c
        )

        # === Step 6: Optional Regularization ===
        reg = (1.0 / sigma_xx + 1.0 / sigma_yy).mean()
        total_loss += self.reg_weight * reg

        if return_individual:
            return total_loss, lv_total.item(), lc_total.item()
        return total_loss
