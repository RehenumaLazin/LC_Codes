import os
import glob
import torch
import rasterio
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Define Lazy Loading Dataset
class FloodDataset(Dataset):
    def __init__(self, input_geotiff_paths, label_geotiff_path):
        """
        Args:
            input_geotiff_paths (list of str): List of 5 paths to input GeoTIFF files.
            label_geotiff_path (str): Path to the label GeoTIFF file.
        """
        self.input_geotiff_paths = input_geotiff_paths
        self.label_geotiff_path = label_geotiff_path

    def __len__(self):
        # Assuming all input GeoTIFFs have the same number of files
        return len(self.input_geotiff_paths[0])

    def __getitem__(self, idx):
        # Read 5 GeoTIFF inputs
        inputs = []
        for path in self.input_geotiff_paths:
            with rasterio.open(path[idx]) as src:
                inputs.append(src.read(1))  # Read the first band

        # Stack the inputs along the channel dimension
        inputs = np.stack(inputs, axis=0)  # Shape: (5, H, W)

        # Read label GeoTIFF
        with rasterio.open(self.label_geotiff_path[idx]) as src:
            label = src.read(1)  # Shape: (H, W)

        # Normalize inputs and labels
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        label = (label > 0).astype(np.float32)

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Define U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, base_channels=32):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(in_channels, base_channels),
            self.conv_block(base_channels, base_channels * 2)
        )
        self.decoder = nn.Sequential(
            self.conv_block(base_channels * 2, base_channels),
            nn.Conv2d(base_channels, out_channels, kernel_size=1)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return torch.sigmoid(dec)

# Define Mean IoU Metric
def compute_mIoU(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
    iou = intersection / (union + 1e-7)
    return iou.mean()

# Training and Validation
def train_and_validate(dataset, k=5, batch_size=8, num_epochs=10, device='cuda'):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    #####
    for fold in range(1,9):
        train_val_indices_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/train_val_indices/train_val_indices_fold_{fold}.json"
        with open(train_val_indices_file, 'r') as f:
            fold_data = json.load(f)
            train_idx = fold_data["train"[train_indices]]
            val_idx = fold_data["validation"[val_indices]]
    
    
    ####

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{k}")

        # Samplers for lazy loading
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        # Initialize Model
        model = UNet(in_channels=5, out_channels=1, base_channels=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scaler = GradScaler()

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            val_mIoU = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    val_mIoU += compute_mIoU(outputs, labels).item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val mIoU: {val_mIoU / len(val_loader):.4f}")

        # Save Model
        torch.save(model.state_dict(), f"unet_model_fold_{fold + 1}.pt")


events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined.csv"  #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'

tiles_df = pd.read_csv(events_file)
tile_names = tiles_df['ras_name'].tolist()  # Assuming the column is named 'tile_name'


DEM = []
LC = []
day1_prec = []
day5_prec = []
day30_prec = []
label_geotiff_path = []
for tile_name in tile_names:
    tile_str = tile_name.split("crop")[0][:-1] 
    DEM.append(f"/p/lustre2/lazin1/cropped_DEM/{tile_str}/dem_{tile_name}")
    LC.append(f"/p/lustre2/lazin1/cropped_LC/{tile_str}/LC_{tile_name}"
    day1_prec.append(f"/p/lustre2/lazin1/1D_prec/{tile_str}/1day_prec_{tile_name}")
    day5_prec.append( f"/p/lustre2/lazin1/5D_prec/{tile_str}/5day_prec_{tile_name}")
    day30_prec.append(f"/p/lustre2/lazin1/30D_prec/{tile_str}/30day_prec_{tile_name}")
    
    label_geotiff_path.append(f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{tile_str}") 
    
    
# GeoTIFF paths

input_geotiff_paths =[
    DEM,
    LC,
    day1_prec,
    day5_prec,
    day30_prec,
                      ]
# input_geotiff_paths = [
#     sorted(glob.glob("/path/to/geotiff/var1/*.tif")),
#     sorted(glob.glob("/path/to/geotiff/var2/*.tif")),
#     sorted(glob.glob("/path/to/geotiff/var3/*.tif")),
#     sorted(glob.glob("/path/to/geotiff/var4/*.tif")),
#     sorted(glob.glob("/path/to/geotiff/var5/*.tif"))
# ]
# label_geotiff_path = sorted(glob.glob("/path/to/labels/*.tif"))

# Dataset
dataset = FloodDataset(input_geotiff_paths, label_geotiff_path)

# Run Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_and_validate(dataset, k=5, batch_size=8, num_epochs=10, device=device)
