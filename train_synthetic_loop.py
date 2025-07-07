import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from model.ctin_model import CTINModel
from loss.multitask_loss import MultiTaskLoss

# === Load synthetic loop dataset ===
X_tensor = torch.load("X_synthetic_noisy.pt")  # [N, 200, 6]
Y_tensor = torch.load("Y_synthetic_noisy.pt")  # [N, 200, 2]

print("Loaded dataset:", X_tensor.shape, Y_tensor.shape)

# === Create dataset and loader ===
dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# === Model setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTINModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=8e-4)
criterion = MultiTaskLoss()

# === Training loop ===
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total_loss, vel_loss, cov_loss = 0.0, 0.0, 0.0

    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        pred_vel, pred_cov = model(X_batch)  # velocity: [B, 200, 2], covariance: [B, 200, 2x2]
        loss, l_vel, l_cov = criterion(pred_vel, pred_cov, Y_batch, return_individual=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        vel_loss += l_vel
        cov_loss += l_cov

    avg_total = total_loss / len(train_loader)
    avg_vel = vel_loss / len(train_loader)
    avg_cov = cov_loss / len(train_loader)
    print(f"Epoch {epoch+1:2d} | Total: {avg_total:.4f} | Vel: {avg_vel:.4f} | Cov: {avg_cov:.4f}")

# === Save model for inference ===
torch.save(model.state_dict(), "ctin_synthetic_loop_noisy.pth")
print("Model saved as ctin_synthetic_loop_noisy.pth")
