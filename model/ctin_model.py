import torch
import torch.nn as nn
from embedding import SpatialEmbedding
from model.temporal_embedding import TemporalEmbedding
from model.decoder import CTINDecoder
from model.output_heads import OutputHeads
from model.spatial_encoder import SpatialEncoder


class CTINModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_heads=4, num_encoder_layers=6, num_decoder_layers=6):
        super(CTINModel, self).__init__()

        self.embedding = SpatialEmbedding(input_dim=input_dim, hidden_dim=hidden_dim)
        self.encoder = SpatialEncoder(dim=hidden_dim, num_heads=num_heads, num_layers=num_encoder_layers)
        self.temporal = TemporalEmbedding(input_dim=input_dim, hidden_dim=hidden_dim)  # uses raw IMU
        self.decoder = CTINDecoder(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_decoder_layers)
        self.output_heads = OutputHeads(hidden_dim=hidden_dim)

    def forward(self, imu_tensor):
        """
        imu_tensor: [B, T, 6] - Raw IMU data (already rotated to navigation frame)
        Returns:
            vel_pred: [B, T, 2]
            cov_pred: [B, T, 3]
        """

        spatial = self.embedding(imu_tensor)        # → [B, T, D]
        encoded = self.encoder(spatial)             # → [B, T, D]
        temporal = self.temporal(imu_tensor)        # → [B, T, D]
        decoded = self.decoder(temporal, encoded)   # → [B, T, D]
        vel_pred, cov_pred = self.output_heads(decoded)

        return vel_pred, cov_pred
