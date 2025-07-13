import torch
import torch.nn as nn
from ctin_project.attention_blocks.temporal_decoder_block import CTINDecoderLayer


class CTINDecoder(nn.Module):
    def __init__(self, input_dim=128, memory_dim=64, num_heads=4, ff_hidden_dim=128, num_layers=6):
        super(CTINDecoder, self).__init__()

        # Project temporal embedding (128) → memory_dim (64)
        self.input_proj = nn.Linear(input_dim, memory_dim)

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            CTINDecoderLayer(dim=memory_dim, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, memory):
        """
        x: temporal embedding [B, T, 128]
        memory: spatial encoder output [B, T, 64]
        """
        x = self.input_proj(x)  # → [B, T, 64]

        for layer in self.layers:
            x = layer(x, memory)

        return x  # [B, T, 64]