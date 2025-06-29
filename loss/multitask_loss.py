import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, learn_uncertainty=True):
        super(MultiTaskLoss, self).__init__()
        if learn_uncertainty:
            self.log_delta_v = nn.Parameter(torch.tensor(0.0))  # log(σ_v)
            self.log_delta_c = nn.Parameter(torch.tensor(0.0))  # log(σ_c)
        else:
            self.register_buffer("log_delta_v", torch.tensor(0.0))
            self.register_buffer("log_delta_c", torch.tensor(0.0))

    def forward(self, vel_pred, cov_pred, vel_gt, pos_gt, dt=0.01):
        B, T, _ = vel_pred.shape

        # === Step 1: Integrate velocity to estimate position ===
        pos_pred = torch.zeros_like(pos_gt)
        pos_pred[:, 0] = vel_pred[:, 0] * dt
        for t in range(1, T):
            pos_pred[:, t] = pos_pred[:, t - 1] + vel_pred[:, t - 1] * dt

        # === Step 2: Compute velocity loss ===
        lv_p = F.mse_loss(pos_pred, pos_gt)
        lv_e = F.mse_loss(vel_pred, vel_gt)
        lv_total = lv_p + lv_e

        # === Step 3: Compute covariance-based NLL loss ===
        sigma_xx = torch.exp(cov_pred[..., 0]) + 1e-6
        sigma_yy = torch.exp(cov_pred[..., 1]) + 1e-6
        sigma_xy = cov_pred[..., 2]

        det = sigma_xx * sigma_yy - sigma_xy ** 2 + 1e-6
        inv_xx = sigma_yy / det
        inv_yy = sigma_xx / det
        inv_xy = -sigma_xy / det

        diff = vel_gt - vel_pred
        dx = diff[..., 0]
        dy = diff[..., 1]

        mahal = inv_xx * dx**2 + 2 * inv_xy * dx * dy + inv_yy * dy**2
        log_det = torch.log(det)
        lc_total = 0.5 * (mahal + log_det).mean()

        # === Step 4: Combine into multi-task loss ===
        delta_v = torch.exp(self.log_delta_v)
        delta_c = torch.exp(self.log_delta_c)

        total_loss = (
            (lv_total / (2 * delta_v**2)) +
            (lc_total / (2 * delta_c**2)) +
            self.log_delta_v + self.log_delta_c
        )

        return total_loss, lv_total.item(), lc_total.item()
