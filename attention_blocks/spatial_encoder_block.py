import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_blocks.local_attention import LocalSelfAttentionBlock
from attention_blocks.global_attention import GlobalSelfAttentionBlock

class SpatialEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.local_block = LocalSelfAttentionBlock(dim)
        self.global_attn = GlobalSelfAttentionBlock(dim, num_heads)
        self.final_proj = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, X, return_attn=False):
        # Local Self-Attention
        Y_local, gamma = self.local_block(X, return_attn=True)

        # Global Self-Attention
        Y_global, weights = self.global_attn(Y_local, return_attn=True)

        # Final 1x1 Conv Projection
        Y_proj = self.final_proj(Y_global.permute(0, 2, 1)).permute(0, 2, 1)

        # Skip connection + activation
        Y_out = F.relu(X + Y_proj)

        return (Y_out, {"local": gamma, "global": weights}) if return_attn else Y_out
