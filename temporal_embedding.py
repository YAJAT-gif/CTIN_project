import torch
import torch.nn as nn


class TemporalEmbedding(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, window_size=200):
        super().__init__()

        # Bi-directional LSTM (output: 2 × hidden_dim)
        self.bilstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Learnable positional encoding for each timestep (0 to window_size-1)
        self.positional_encoding = nn.Embedding(window_size, hidden_dim * 2)

        # Save window size for indexing
        self.window_size = window_size

    def forward(self, x):
        """
        x: IMU sequence tensor of shape [B, T, input_dim]
        Returns: temporally encoded sequence [B, T, 2 * hidden_dim]
        """
        B, T, _ = x.size()
        assert T <= self.window_size, f"Input sequence length {T} exceeds max window size {self.window_size}"

        # BiLSTM encoding
        lstm_out, _ = self.bilstm(x)  # [B, T, 2 * hidden_dim]

        # Create timestep indices: [0, 1, ..., T-1]
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)  # [B, T]
        pe = self.positional_encoding(positions)  # [B, T, 2 * hidden_dim]

        # Add positional encoding to BiLSTM output
        return lstm_out + pe