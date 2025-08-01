import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ctin_project.model.ctin_model import CTINModel
from ctin_project.loss.velocity_loss import VelocityOnlyLoss
from ctin_project.sequence_window_dataset import SequenceWindowDataset

# Config
csv_dir = "../ctin_csv_output"  # Folder containing all ctin_dataset_*.csv files
window_size = 200
stride = 10
batch_size = 64
num_epochs = 25
learning_rate = 1e-4
save_path = "../ctin_model_tlio_GRU_highStride.pth"

# Dataset and DataLoader
dataset = SequenceWindowDataset(csv_dir, window_size=window_size, stride=stride)
print("Dataset size (windows):", len(dataset))

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Num batches per epoch:", len(train_loader))

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model, loss, optimizer
model = CTINModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = VelocityOnlyLoss()


# Training loop
start_time = time.time()
for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()

    total_loss = 0.0
    vel_loss = 0.0
    cov_loss = 0.0

    for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        pred_vel, pred_cov = model(X_batch)

        loss, l_vel, l_cov = criterion(pred_vel, pred_cov, Y_batch, return_individual=True)
        if torch.isnan(pred_vel).any():
            print("NaN detected in pred_vel!")
            print("Batch idx:", batch_idx)
            print("Input stats:", X_batch.mean(), X_batch.std())
            print("Vel stats:", pred_vel.min(), pred_vel.max(), pred_vel.mean(), pred_vel.std())
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
        optimizer.step()

        total_loss += loss.item()
        vel_loss += l_vel
        cov_loss += l_cov

    avg_total = total_loss / len(train_loader)
    avg_vel = vel_loss / len(train_loader)
    avg_cov = cov_loss / len(train_loader)
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    remaining = epoch_time * (num_epochs - epoch - 1)

    print(f"Epoch {epoch+1:2d} | Total: {avg_total:.4f} | Vel: {avg_vel:.4f} | Cov: {avg_cov:.4f} | "
          f"Time: {epoch_time:.1f}s | ETA: {remaining/60:.1f} min")

# Save final model
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Model saved to: {save_path}")
