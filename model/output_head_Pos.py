import torch
import torch.nn as nn

class OutputHeads(nn.Module):
    def __init__(self, hidden_dim=64):
        super(OutputHeads, self).__init__()

        # Displacement head (predicts per-timestep relative motion)
        self.disp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        # Anchor velocity head (predicts initial velocity at window start)
        self.anchor_vel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, h):
        """
        Args:
            h (Tensor): [B, T, D] – decoder output
        Returns:
            disp: [B, T, 2] – relative displacements
            anchor_vel: [B, 2] – predicted initial velocity at anchor point
        """
        disp = self.disp_head(h)  # [B, T, 2]
        anchor_feat = h[:, 0, :]  # [B, D] -> single time step feature
        anchor_vel = self.anchor_vel_head(anchor_feat)  # [B, 2]

        return disp, anchor_vel
