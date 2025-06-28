import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalSelfAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.key_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.value_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.att_score = nn.Conv1d(2 * dim, dim, kernel_size=1)
        self.after_local_conv = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, X, return_attn=False):
        B, T, D = X.shape
        X_perm = X.permute(0, 2, 1)                    # [B, D, T]

        K = self.key_conv(X_perm).permute(0, 2, 1)     # [B, T, D]
        Q = X                                          # [B, T, D]
        V = self.value_conv(X_perm).permute(0, 2, 1)   # [B, T, D]

        QK = torch.cat([Q, K], dim=-1).permute(0, 2, 1)  # [B, 2D, T]
        gamma = F.relu(self.att_score(QK)).permute(0, 2, 1)  # [B, T, D]

        C2 = gamma * V
        Y_local = K + C2
        Y_proj = self.after_local_conv(Y_local.permute(0, 2, 1)).permute(0, 2, 1)

        return (Y_proj, gamma) if return_attn else Y_proj
