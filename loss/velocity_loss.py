import torch
import torch.nn as nn
import torch.nn.functional as F

class VelocityOnlyLoss(nn.Module):
    def __init__(self):
        super(VelocityOnlyLoss, self).__init__()

    def forward(self, vel_pred, cov_pred, vel_gt, return_individual=False):
        vel_loss = F.mse_loss(vel_pred, vel_gt)
        if return_individual:
            return vel_loss, vel_loss.item(), 0.0  # total, vel, dummy_cov
        return vel_loss
