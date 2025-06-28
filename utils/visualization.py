import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import numpy as np


def plot_global_attention(weights, layer_idx=0, head=0):
    """
    Plot global attention heatmap for one head of a specific layer.
    """
    attn = weights[layer_idx]["global"][0, head].detach().cpu().numpy()
    sns.heatmap(attn, cmap="viridis")
    plt.title(f"Global Attention Heatmap - Layer {layer_idx + 1}, Head {head}")
    plt.xlabel("Key Timestep")
    plt.ylabel("Query Timestep")
    plt.show()


def plot_local_attention(gamma, layer_idx=0):
    """
    Plot mean local attention intensity per timestep.
    """
    attn = gamma[layer_idx]["local"][0].mean(dim=-1).detach().cpu().numpy()
    plt.plot(attn)
    plt.title(f"Local Attention Intensity - Layer {layer_idx + 1}")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Activation")
    plt.grid(True)
    plt.show()


def compare_pca(before_tensor, after_tensor, title_suffix=""):
    """
    Compare PCA of embedding before and after attention.
    """
    before = before_tensor[0].detach().cpu().numpy()
    after = after_tensor[0].detach().cpu().numpy()

    pca = PCA(n_components=2)
    before_pca = pca.fit_transform(before)
    after_pca = pca.fit_transform(after)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(before_pca[:, 0], before_pca[:, 1], c=np.arange(len(before_pca)), cmap="viridis")
    plt.title(f"PCA Before Encoder {title_suffix}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.scatter(after_pca[:, 0], after_pca[:, 1], c=np.arange(len(after_pca)), cmap="viridis")
    plt.title(f"PCA After Encoder {title_suffix}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def plot_temporal_embedding_heatmap(temporal_output):
    """
    Show [features x time] heatmap of the embedding
    """
    data = temporal_output[0].detach().cpu().numpy().T  # [128, 200]
    sns.heatmap(data, cmap="viridis")
    plt.title("Temporal Embedding Heatmap")
    plt.xlabel("Timestep")
    plt.ylabel("Feature Index")
    plt.tight_layout()
    plt.show()

def plot_temporal_embedding_pca(temporal_output):
    """
    PCA of the embedding sequence: each timestep projected to 2D
    """
    from sklearn.decomposition import PCA

    data = temporal_output[0].detach().cpu().numpy()  # [200, 128]
    pca = PCA(n_components=2)
    pca_out = pca.fit_transform(data)

    plt.figure(figsize=(6, 5))
    plt.scatter(pca_out[:, 0], pca_out[:, 1], c=np.arange(200), cmap="viridis")
    plt.title("PCA of Temporal Embedding")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Timestep")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
