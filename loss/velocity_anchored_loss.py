import torch
import torch.nn as nn

class VelocityAnchoredLoss(nn.Module):
    def __init__(self, dt=0.05, lambda_anchor=1.0):
        super(VelocityAnchoredLoss, self).__init__()
        self.dt = dt
        self.lambda_anchor = lambda_anchor
        self.mse = nn.MSELoss()

    def forward(self, pred_disp, pred_anchor_vel, gt_disp, gt_anchor_pos):
        """
        Args:
            pred_disp:       [B, T, 2] – predicted displacements
            pred_anchor_vel: [B, 2]   – predicted anchor velocity
            gt_disp:         [B, T, 2] – ground truth displacements
            gt_anchor_pos:   [B, 2]   – ground truth anchor position
        Returns:
            total_loss: scalar
        """
        # === Displacement loss ===
        loss_disp = self.mse(pred_disp, gt_disp)

        # === Anchor position from integrated velocity ===
        window_len = pred_disp.size(1)
        pred_anchor_pos = pred_anchor_vel * self.dt * window_len  # [B, 2]

        # === Anchor supervision ===
        loss_anchor = self.mse(pred_anchor_pos, gt_anchor_pos)

        total_loss = loss_disp + self.lambda_anchor * loss_anchor
        return total_loss
