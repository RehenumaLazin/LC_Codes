import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Import model architectures ---
from models import LocalUNetLightweight, LocalUNetAttention, GlobalEncoder  # Assume you saved them in models.py

# --- Dataset loader stub (you should implement your own) ---
class FloodDataset(torch.utils.data.Dataset):
    def __init__(self, local_inputs, global_inputs, targets):
        self.local_inputs = local_inputs
        self.global_inputs = global_inputs
        self.targets = targets

    def __len__(self):
        return len(self.local_inputs)

    def __getitem__(self, idx):
        return {
            'local_input': self.local_inputs[idx],     # shape: [C, H, W]
            'global_input': self.global_inputs[idx],   # shape: [C, h, w]
            'target': self.targets[idx]                # shape: [1, H, W]
        }

# --- Select model type ---
def get_model(model_type="lightweight", in_channels=5, global_channels=64):
    if model_type == "lightweight":
        model = LocalUNetLightweight(in_channels, global_channels)
    elif model_type == "attention":
        model = LocalUNetAttention(in_channels, global_channels)
    else:
        raise ValueError("Model type must be 'lightweight' or 'attention'")
    return model

# --- Evaluate the model ---
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            li = batch['local_input'].to(device)
            target = batch['target'].to(device)
            pred = model(li)
            loss = criterion(pred, target)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# --- Visualize prediction vs ground truth ---
def visualize_predictions(model, dataloader, device, num_samples=3):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            li = batch['local_input'].to(device)
            target = batch['target'].to(device)
            pred = model(li)

            # Visualize some samples
            for j in range(num_samples):
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(li[j, 0].cpu().numpy(), cmap='Blues')  # Local input channel 0
                axes[0].set_title('Local Input')
                axes[1].imshow(target[j, 0].cpu().numpy(), cmap='Blues')  # Ground truth
                axes[1].set_title('Ground Truth')
                axes[2].imshow(pred[j, 0].cpu().numpy(), cmap='Blues')  # Prediction
                axes[2].set_title('Prediction')
                plt.show()

            if i >= num_samples:
                break

# --- Training Step ---
def train_model(model_type="lightweight",
                epochs=50,
                lr=1e-4,
                batch_size=4,
                checkpoint_path="checkpoints/",
                use_global_supervision=False):

    # Simulate dummy data (replace with real dataset loader)
    local_input = torch.randn(100, 5, 256, 256)  # 100 samples
    global_input = torch.randn(100, 1, 32, 32)   # Coarse global context
    target = torch.randn(100, 1, 256, 256)       # Ground truth
    global_target = torch.randn(100, 64, 32, 32) if use_global_supervision else None

    dataset = FloodDataset(local_input, global_input, target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    global_encoder = GlobalEncoder(in_channels=1, out_channels=64)
    local_model = get_model(model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_encoder.to(device)
    local_model.to(device)

    # Losses and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(local_model.parameters()) + list(global_encoder.parameters()), lr=lr)

    # Checkpoint loading
    start_epoch = 0
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_file = os.path.join(checkpoint_path, f"{model_type}_checkpoint.pth")
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        local_model.load_state_dict(checkpoint['local_model_state'])
        global_encoder.load_state_dict(checkpoint['global_encoder_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        local_model.train()
        global_encoder.train()
        total_loss = 0

        for batch in dataloader:
            li = batch['local_input'].to(device)
            gi = batch['global_input'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()

            global_features = global_encoder(gi)
            pred = local_model(li, global_features)

            loss = criterion(pred, target)

            if use_global_supervision and global_target is not None:
                gt_global = global_target.to(device)
                loss += 0.2 * criterion(global_features, gt_global)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

        # Evaluate the model on the validation set
        val_loss = evaluate_model(local_model, dataloader, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'local_model_state': local_model.state_dict(),
            'global_encoder_state': global_encoder.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, checkpoint_file)

        # Visualization (you can adjust the number of samples to visualize)
        visualize_predictions(local_model, dataloader, device, num_samples=3)

# Example usage:
train_model(model_type="lightweight", epochs=10, lr=1e-4, batch_size=4, checkpoint_path="checkpoints/", use_global_supervision=True)
