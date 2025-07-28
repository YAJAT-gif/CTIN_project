import torch.nn as nn
import torch.nn.functional as F

class SpatialEmbedding(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [batch_size, window_size, input_dim]
        x = x.permute(0, 2, 1)       # → [batch_size, input_dim, window_size]
        x = self.conv1d(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)       # → [batch_size, window_size, hidden_dim]
        x = self.linear(x)           # Linear layer per timestep
        return x
