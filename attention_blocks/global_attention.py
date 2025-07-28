import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GlobalSelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=16):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, X, return_attn=False):
        B, T, D = X.shape
        X_perm = X.permute(0, 2, 1)  # [B, D, T]

        Q = self.q_proj(X_perm).permute(0, 2, 1)  # [B, T, D]
        K = self.k_proj(X_perm).permute(0, 2, 1)
        V = self.v_proj(X_perm).permute(0, 2, 1)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, d]
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]
        weights = F.softmax(scores, dim=-1)

        context = torch.matmul(weights, V)  # [B, H, T, d]
        context = context.transpose(1, 2).contiguous().view(B, T, D)

        out = self.out_proj(context.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T, D]

        return (out, weights) if return_attn else out
