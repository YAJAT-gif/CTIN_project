import torch
import torch.nn as nn

class OutputHeads(nn.Module):
    def __init__(self, hidden_dim=64):
        super(OutputHeads, self).__init__()
        # Velocity head: Predicts [vx, vy]
        self.vel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        # Covariance head: Predicts [log(σ_xx), log(σ_yy)]
        self.cov_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Diagonal only
        )

        # Optional: initialize biases to small positive log-variances
        nn.init.constant_(self.cov_head[-1].bias, 0.0)

    def forward(self, h):
        """
        Args:
            h (Tensor): [B, T, D] – decoder output
        Returns:
            vel: [B, T, 2]
            cov: [B, T, 2] – log variances
        """
        vel = self.vel_head(h)
        cov = self.cov_head(h)
        return vel, cov
