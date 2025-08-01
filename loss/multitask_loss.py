import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, learn_uncertainty=True, min_variance=1e-2, reg_weight=0.01):
        super(MultiTaskLoss, self).__init__()
        self.min_variance = min_variance
        self.reg_weight = reg_weight

        if learn_uncertainty:
            self.log_delta_p = nn.Parameter(torch.tensor(0.0))  # log(σ_pos)
            self.log_delta_c = nn.Parameter(torch.tensor(0.0))  # log(σ_cov)
        else:
            self.register_buffer("log_delta_p", torch.tensor(0.0))
            self.register_buffer("log_delta_c", torch.tensor(0.0))

    def forward(self, pos_pred, cov_pred, pos_gt, return_individual=False):
        # === Position Loss ===
        lp_total = F.mse_loss(pos_pred, pos_gt)

        # === Covariance Loss (Negative Log-Likelihood) ===
        sigma_xx = torch.clamp(torch.exp(cov_pred[..., 0]), min=self.min_variance)
        sigma_yy = torch.clamp(torch.exp(cov_pred[..., 1]), min=self.min_variance)

        det = sigma_xx * sigma_yy
        inv_xx = 1.0 / sigma_xx
        inv_yy = 1.0 / sigma_yy

        diff = pos_gt - pos_pred
        dx = diff[..., 0]
        dy = diff[..., 1]

        mahal = inv_xx * dx**2 + inv_yy * dy**2
        log_det = torch.log(det + 1e-6)
        lc_total = 0.5 * (mahal + log_det).mean()

        # === Combine Multi-task Loss ===
        delta_p = torch.exp(self.log_delta_p)
        delta_c = torch.exp(self.log_delta_c)

        total_loss = (
            (lp_total / (2 * delta_p**2)) +
            (lc_total / (2 * delta_c**2)) +
            self.log_delta_p + self.log_delta_c
        )

        # === Optional Covariance Regularization ===
        reg = (1.0 / sigma_xx + 1.0 / sigma_yy).mean()
        total_loss += self.reg_weight * reg

        if return_individual:
            return total_loss, lp_total.item(), lc_total.item()
        return total_loss
