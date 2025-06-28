import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CTINDecoderLayer(nn.Module):
    def __init__(self, dim=64, num_heads=4, ff_hidden_dim=128):
        super(CTINDecoderLayer, self).__init__()

        # Masked Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        # Cross-Attention (Decoder-Encoder)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        # Feedforward sub-layer
        self.feedforward = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, dim)
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, memory):
        """
        x: decoder input [batch, seq_len, dim]
        memory: encoder output [batch, seq_len, dim]
        """
        B, T, D = x.shape

        # 1. Masked self-attention
        tgt_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)  # [T, T]
        attn_out1, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + attn_out1)

        # 2. Multi-head attention over encoder output
        attn_out2, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + attn_out2)

        # 3. Feedforward + Residual
        ff_out = self.feedforward(x)
        x = self.norm3(x + ff_out)

        return x
