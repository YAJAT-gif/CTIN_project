import torch
from data_loader import load_and_process_data
from embedding import SpatialEmbedding
from temporal_embedding import TemporalEmbedding
from model.spatial_encoder import SpatialEncoder
from utils.visualization import plot_global_attention, plot_local_attention, compare_pca
from utils.visualization import (
    plot_temporal_embedding_heatmap,
    plot_temporal_embedding_pca
)

# ---------- CONFIGURATION ----------
IMU_PATH = "imu_data.csv"
GT_PATH = "gt_data.csv"
WINDOW_SIZE = 200
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 6

# ---------- LOAD & PROCESS DATA ----------
X_data, Y_data = load_and_process_data(IMU_PATH, GT_PATH, window_size=WINDOW_SIZE)
X_tensor = torch.tensor(X_data, dtype=torch.float32)

# ---------- EMBEDDING ----------
embedder = SpatialEmbedding(input_dim=6, hidden_dim=HIDDEN_DIM)
embedded_output = embedder(X_tensor)  # [batch, 200, 64]

temporal_encoder = TemporalEmbedding(input_dim=6, hidden_dim=64, window_size=200)
temporal_output = temporal_encoder(X_tensor[:10])  # [10, 200, 128]

# ---------- ENCODER STACK ----------
encoder_stack = SpatialEncoder(dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)
stacked_output, attn_logs = encoder_stack(embedded_output[:10], return_attn=True)

# ---------- VISUALIZATION ----------
# Plot attention for Layer 1 and Layer 6
plot_global_attention(attn_logs, layer_idx=0, head=0)
plot_global_attention(attn_logs, layer_idx=5, head=0)
plot_local_attention(attn_logs, layer_idx=0)
plot_local_attention(attn_logs, layer_idx=5)

# PCA visualization
compare_pca(embedded_output, stacked_output, title_suffix=f"(Encoder Layers: {NUM_LAYERS})")
temporal_output = temporal_encoder(X_tensor[:1])
plot_temporal_embedding_heatmap(temporal_output)
plot_temporal_embedding_pca(temporal_output)