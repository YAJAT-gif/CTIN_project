import torch.nn as nn
from ctin_project.attention_blocks.spatial_encoder_block import SpatialEncoderBlock

class SpatialEncoder(nn.Module):
    def __init__(self, dim=64, num_heads=4, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            SpatialEncoderBlock(dim=dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, return_attn=False):
        attn_logs = [] if return_attn else None

        for layer in self.layers:
            if return_attn:
                x, attn = layer(x, return_attn=True)
                attn_logs.append(attn)
            else:
                x = layer(x)

        return (x, attn_logs) if return_attn else x