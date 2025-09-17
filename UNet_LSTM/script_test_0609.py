import os
import argparse
import glob
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import rasterio
from rasterio.enums import Resampling
import glob
from collections import defaultdict
from sklearn.model_selection import KFold
import torch.nn.functional as F



# --- Dataset ---
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
import numpy as np
import pandas as pd
import torch
torch.cuda.empty_cache()

from torch.utils.data import Dataset
import rasterio
from torch.utils.data.sampler import SubsetRandomSampler



import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio

# class FloodDataset(Dataset):
#     def __init__(self, dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, hours, seq_len,
#                  DEM_mean, DEM_std, precip_mean, precip_std,
#                  flood_mean=None, flood_std=None, return_metadata=False, mode='train'):
        
#         self.dem_dir = dem_dir
#         self.lc_dir = lc_dir
#         self.precip_dir = precip_dir
#         self.flood_dir = flood_dir
#         self.seq_len = seq_len
#         self.hours = hours
#         self.return_metadata = return_metadata
#         self.mode = mode.lower()

#         self.DEM_mean = DEM_mean
#         self.DEM_std = DEM_std
#         self.precip_mean = precip_mean
#         self.precip_std = precip_std
#         self.flood_mean = flood_mean
#         self.flood_std = flood_std

#         self.tiles = pd.read_csv(tile_csv)['ras_name'].tolist()
#         self.samples = self._generate_samples()

#     def _generate_samples(self):
#         samples = []
#         for tile in self.tiles:
#             event = "event2"
#             for i in range(self.seq_len - 1, len(self.hours)):
#                 seq_hours = self.hours[i - self.seq_len + 1:i + 1]
#                 target_hour = self.hours[i]
#                 samples.append((tile, seq_hours, target_hour))
#         return samples

#     def _read_raster(self, path):
#         with rasterio.open(path) as src:
#             array = src.read(1).astype(np.float32)
#             transform = src.transform
#             crs = src.crs
#         return array, transform, crs

#     def _load_static_inputs(self, tile):
#         dem_path = os.path.join(self.dem_dir, tile)
#         lc_path = os.path.join(self.lc_dir, f"LC_{tile}")

#         dem, _, _ = self._read_raster(dem_path)
#         lc, _, _ = self._read_raster(lc_path)

#         dem_norm = (dem - self.DEM_mean) / self.DEM_std
#         lc_binary = np.where(lc == 11, 1, 0).astype(np.float32)

#         static_input = np.stack([dem_norm, lc_binary], axis=0)  # Shape: (2, H, W)
#         return torch.from_numpy(static_input)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         tile, seq_hours, target_hour = self.samples[idx]
#         event = "event2"

#         static_input = self._load_static_inputs(tile)  # Lazy load

#         # Load precipitation sequence (T, H, W)
#         precip_seq = []
#         for h in seq_hours:
#             p_path = os.path.join(self.precip_dir, f"1H_prec_{event}_{h.strftime('%Y%m%d%H')}_{tile}")
#             p, _, _ = self._read_raster(p_path)
#             p_norm = (p - self.precip_mean) / self.precip_std
#             precip_seq.append(p_norm)
#         precip_tensor = torch.from_numpy(np.stack(precip_seq, axis=0).astype(np.float32))

#         # Load flood depth label (1, H, W)
#         label_path = os.path.join(self.flood_dir, f"{event}_{target_hour.strftime('%Y%m%d%H')}_{tile}")
#         label, transform, crs = self._read_raster(label_path)

#         if self.flood_mean is not None and self.flood_std is not None and self.mode == 'train':
#             label = (label - self.flood_mean) / self.flood_std

#         label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

#         if self.return_metadata:
#             return static_input, precip_tensor, label_tensor, transform, crs
#         return static_input, precip_tensor, label_tensor



class FloodDataset(Dataset):
    def __init__(self, dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, hours, seq_len,
                 DEM_mean, DEM_std, precip_mean, precip_std,
                 flood_mean=None, flood_std=None, return_metadata=False, mode='train'):
        
        self.dem_dir = dem_dir
        self.lc_dir = lc_dir
        self.precip_dir = precip_dir
        self.flood_dir = flood_dir
        self.seq_len = seq_len
        self.hours = hours
        self.return_metadata = return_metadata
        self.mode = mode.lower()

        self.DEM_mean = DEM_mean
        self.DEM_std = DEM_std
        self.precip_mean = precip_mean
        self.precip_std = precip_std
        self.flood_mean = flood_mean
        self.flood_std = flood_std

        self.tiles = pd.read_csv(tile_csv)['ras_name'].tolist()
        self.samples = self._generate_samples()

        # # Cache static inputs to reduce disk I/O
        # self.static_inputs = {}
        # for tile in self.tiles:
        #     self.static_inputs[tile] = self._load_static_inputs(tile)

    def _generate_samples(self):
        samples = []
        for tile in self.tiles:
            event = "event2"
            for i in range(self.seq_len - 1, len(self.hours)):
                seq_hours = self.hours[i - self.seq_len + 1:i + 1]
                target_hour = self.hours[i]
                # if all(os.path.exists(os.path.join(self.precip_dir, f"1H_prec_{event}_{h.strftime('%Y%m%d%H')}_{tile}")) for h in seq_hours) and \
                #    os.path.exists(os.path.join(self.flood_dir, f"{event}_{target_hour.strftime('%Y%m%d%H')}_{tile}")):
                #     samples.append((tile, seq_hours, target_hour))
                samples.append((tile, seq_hours, target_hour))
        return samples

    def _read_raster(self, path):
        with rasterio.open(path) as src:
            array = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
        return array, transform, crs

    def _load_static_inputs(self, tile):
        dem_path = os.path.join(self.dem_dir, f"{tile}")
        # print(dem_path, os.path.exists(dem_path))
        lc_path = os.path.join(self.lc_dir, f"LC_{tile}")
        # print(lc_path, os.path.exists(lc_path))
        

        dem, _, _ = self._read_raster(dem_path)
        lc, _, _ = self._read_raster(lc_path)

        dem_norm = (dem - self.DEM_mean) / self.DEM_std
        lc_binary = np.where(lc == 11, 1, 0).astype(np.float32)

        # Shape: (2, H, W)
        static_input = np.stack([dem_norm, lc_binary], axis=0)
        return torch.from_numpy(static_input)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tile, seq_hours, target_hour = self.samples[idx]
        event = "event2"
        static_input = self._load_static_inputs(tile)

        # static_input = self.static_inputs[tile]  # Already (2, H, W)

        # Load precipitation sequence (T, H, W)
        precip_seq = []
        for h in seq_hours:
            p_path = os.path.join(self.precip_dir, f"1H_prec_{event}_{h.strftime('%Y%m%d%H')}_{tile}")
            # print(p_path, os.path.exists(p_path))
            p, _, _ = self._read_raster(p_path)
            p_norm = (p - self.precip_mean) / self.precip_std
            precip_seq.append(p_norm)
        precip_seq = np.stack(precip_seq, axis=0).astype(np.float32)
        precip_tensor = torch.from_numpy(precip_seq)

        # Load label
        label_path = os.path.join(self.flood_dir, f"{event}_{target_hour.strftime('%Y%m%d%H')}_{tile}")
        # print(label_path, os.path.exists(label_path))
        label, transform, crs = self._read_raster(label_path)

        if self.flood_mean is not None and self.flood_std is not None and self.mode == 'train':
            label = (label - self.flood_mean) / self.flood_std

        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        if self.return_metadata:
            return static_input, precip_tensor, label_tensor, transform, crs
        return static_input, precip_tensor, label_tensor


# class StaticFloodDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
#     def __len__(self): return len(self.data)
#     def __getitem__(self, idx): return self.data[idx]
    
class StaticFloodDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx]

    @property
    def samples(self):
        return self.data
# --- Model ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)

class UNetLSTM(nn.Module):
    def __init__(self, in_static_ch=2, in_seq_ch=1, seq_len=10, hidden_dim=64):
        super().__init__()
        self.encoder_static = ConvBlock(in_static_ch, 32)
        self.encoder_seq = ConvBlock(in_seq_ch, 32)
        self.pool = nn.MaxPool2d(2)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(32 + hidden_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, static_input, seq_input):
        # B, T, _, H, W = seq_input.shape
        B, T, H, W = seq_input.shape
        x_static = self.encoder_static(static_input)  # (B, 32, H, W)
        x_seq = seq_input.view(B * T, 1, H, W)
        x_seq = self.encoder_seq(x_seq)
        x_seq = self.pool(x_seq)
        x_seq = x_seq.view(B, T, 32, H//2, W//2)
        x_seq = x_seq.mean(dim=[3, 4])
        lstm_out, _ = self.lstm(x_seq)
        lstm_feat = lstm_out[:, -1, :].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined = torch.cat([x_static, lstm_feat], dim=1)
        # return self.decoder(combined).squeeze(1)
        return self.decoder(combined)

# --- Training ---
def train(model, train_loader, val_loader, optimizer, criterion, checkpoint_path, scaler, fold_idx, start_epoch, num_epochs=10):
    model.train()
    running_loss = 0.0
    train_losses, val_losses, mse = [], [], []
    # start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    # if start_epoch > num_epochs:
    #     start_epoch = 0
    # print(start_epoch)
    
    for epoch in range(start_epoch, num_epochs):
        for static, seq, target in tqdm(train_loader.dataset, desc=f"Epoch {epoch}"):


        # for inputs, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
        #     static = inputs[:, :2] 
        #     seq = inputs[:, 2:].unsqueeze(2) 
            static, seq, target = static.to(device), seq.to(device), target.to(device)
            # seq = seq.unsqueeze(2)
            # print(seq.shape, static.shape,target.shape )
            if seq.dim() == 3:  # (T, 1, H, W)
                seq = seq.unsqueeze(0)  # → (1, T, 1, H, W)
            if static.dim() == 3:
                static = static.unsqueeze(0)  # → (1, 2, H, W)
            # if target.dim() == 3:
            #     target = target.unsqueeze(0)  # → (1, 1, H, W)
                
            # print(seq.shape, static.shape,target.shape )
            optimizer.zero_grad()
            
            with autocast():
                output = model(static, seq)
                output = output.squeeze(1)
                loss = criterion(output, target)

                # Backward pass with scaled gradients
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

                # running_loss += loss.item() * inputs.size(0)
                
            if loss is not None and torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * seq.size(0)
            else:
                print(f"Skipping step due to invalid loss at epoch {epoch}")
            
            
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")
        save_checkpoint(model, optimizer, epoch,fold_idx, checkpoint_path)
        
        
        model.eval()
        val_loss = 0.0
        mse_score = 0.0
        
        with torch.no_grad():
            # for inputs, target in val_loader:
                # static = inputs[:, :2] 
                # seq = inputs[:, 2:].unsqueeze(2) 
            for static, seq, target in val_loader.dataset:
                # seq = seq.unsqueeze(2)
                
                if seq.dim() == 3:  # (T, 1, H, W)
                    seq = seq.unsqueeze(0)  # → (1, T, 1, H, W)
                if static.dim() == 3:
                    static = static.unsqueeze(0)  # → (1, 2, H, W)
                # if target.dim() == 3:
                #     target = target.unsqueeze(0)  # → (1, 1, H, W)
                static, seq, target = static.to(device), seq.to(device), target.to(device)
                with autocast():
                    output = model(static, seq)
                    output = output.squeeze(1)
                    
                    loss = criterion(output, target)
                    
                val_loss += loss.item() * seq.size(0)
                mse_score += MSE(output, target)
            
            
        

        val_loss /= len(val_loader.dataset)
        # iou_score /= len(val_loader)
        mse_score /= len(val_loader)
        val_losses.append(val_loss)
        # miou.append(iou_score)
        mse.append(mse_score)
        # print(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {iou_score:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation MSE: {mse_score:.4f}")
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch,fold_idx, checkpoint_path)
        print(checkpoint_path, "saved")
    return model, train_losses, val_losses, mse

# --- Testing ---
def test(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, target in tqdm(dataloader, desc="Testing"):
            static = inputs[:, :2] 
            seq = inputs[:, 2:].unsqueeze(2) 
            static, seq, target = static.to(device), seq.to(device), target.to(device)
            output = model(static, seq)
            loss = criterion(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")
    
    
# Load the saved model for testing
def load_model(model_path, device='cuda'):
    model = UNetLSTM()
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    return model




def save_checkpoint(model, optimizer, epoch, fold, checkpoint_path):
    """
    Save model, optimizer state, epoch, and fold to a checkpoint.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer to save.
        epoch: The current epoch.
        fold: The current fold.
        checkpoint_path: Path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'fold': fold,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at fold {fold}, epoch {epoch} to {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Load model, optimizer state, epoch, and fold from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint.
        model: The PyTorch model to load state into.
        optimizer: The optimizer to load state into.

    Returns:
        start_epoch: The epoch to resume training from.
        start_fold: The fold to resume training from.
    """
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        start_fold = checkpoint['fold'] 
        print(f"Resumed training from fold {start_fold}, epoch {start_epoch}")
        return start_epoch, start_fold
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0, 0

def custom_collate_fn(batch):
    # print(len(batch[0]))
    """
    Custom collate function for training and testing.

    For training:
        Returns only inputs and labels.

    For testing:
        Returns inputs, transforms, and CRS metadata.
    """
    if len(batch[0]) == 2:  # Training mode
        inputs = torch.stack([item[0] for item in batch])  # Stack inputs
        labels = torch.stack([item[1] for item in batch])  # Stack labels
        return inputs, labels
    elif len(batch[0]) == 4:  # Testing mode
        inputs = torch.stack([item[0] for item in batch])  # Stack inputs
        transforms = [item[1] for item in batch]  # Extract transforms
        crs_list = [item[2] for item in batch]  # Extract CRS
        return inputs, transforms, crs_list

def MSE(pred, target): #(label - np.mean(FD_mean)) / np.mean(FD_std)
    mse = F.mse_loss(pred, target, reduction='mean').item()

    
    return mse
def kfold_train(dataset, k_folds, batch_size, model_dir, num_epochs):
    model = UNetLSTM()
    model = nn.DataParallel(model).to(device)
    checkpoint_path = os.path.join(model_dir, "model_checkpoint.pth")
    
    # Extract all unique days (dates only)
    all_dates = sorted(list({s[2].date() for s in dataset.samples}))
    # print("Sample entry example:", dataset.samples[0])
    # print("Type of s[2]:", type(dataset.samples[0][2]))
    # print(dataset.samples)
    # print(len(all_dates),all_dates,  {s[2].date() for s in dataset.samples} )
    assert len(all_dates) >= k_folds, "Not enough unique days for the requested number of folds."
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    

    criterion = nn.MSELoss()
    
    
    start_epoch, start_fold = load_checkpoint(checkpoint_path, model, optimizer)

    fold_losses = []

    for fold_idx in range(start_fold, k_folds):
        val_day = all_dates[fold_idx]

        train_indices = [i for i, s in enumerate(dataset.samples) if s[2].date() != val_day]
        val_indices = [i for i, s in enumerate(dataset.samples) if s[2].date() == val_day]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, collate_fn=custom_collate_fn, pin_memory=True)
        print("train_loader", len(train_loader.dataset), len(train_loader.dataset[0]))
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, collate_fn=custom_collate_fn, pin_memory=True)

        model = UNetLSTM()
        model = nn.DataParallel(model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = GradScaler()
        criterion = nn.MSELoss()
        
        model_save_path = os.path.join(model_dir, f"unet_model_fold_{fold_idx}.pt") #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/unet_model_fold_{fold + 1}.pt"

        if fold_idx > 0 and os.path.exists(model_save_path):
            
            model = load_model(model_save_path)
            model = nn.DataParallel(model).to(device)

        print(f"Starting Fold {fold_idx+1}/{k_folds}, Validation Day: {val_day}")

        # for epoch in range(start_epoch + 1, num_epochs + 1):
        trained_model, train_losses, val_losses, mse = train(model, train_loader,val_loader, optimizer, criterion, checkpoint_path, scaler, fold_idx, start_epoch, num_epochs=num_epochs)
            # test(model, val_loader, criterion)

        # Save the model after each fold
        fold_losses.append((train_losses, val_losses))
        model_save_path = os.path.join(model_dir, f"unet_model_fold_{fold_idx+1}.pt")
        torch.save(trained_model.state_dict(), model_save_path) #torch.save(model.state_dict(), str(model_path))
        # np.save(f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/train_val_losses_fold_{fold + 1}.npy" , fold_losses)
        np.save(os.path.join(model_dir, f"train_val_losses_fold_{fold_idx + 1}.npy") , fold_losses)
        # np.save(f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/val_miou_{fold + 1}.npy" , miou)
        np.save(os.path.join(model_dir, f"val_mse_{fold_idx + 1}.npy"),mse)
        print(f"Model for fold {fold_idx + 1} saved as {model_save_path}.\n")
        start_epoch = 0


    
    

# event_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/event2"
events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_DEM_tiles.csv" #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma_hourly_event2.csv"#f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma_hourly_event2_test.csv"
tile_csv = events_file

model_dir = f"/p/vast1/lazin1/UNet_trains/Sonoma_event2_output_prec_dem_LC_HOURLY_UNET_LSTM"
os.makedirs(model_dir, exist_ok=True)



# tiles_df = pd.read_csv(events_file)
# tile_names = tiles_df['ras_name'].tolist()


dem_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma"
lc_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_LC/Sonoma"
# flood_paths.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/{event}/{tile_name}")

precip_dir = "/p/vast1/lazin1/UNet_inputs/Geotiff_var/1H_prec/Sonoma/event2"
flood_dir = "/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/event2"

# --- Time setup ---
sequence_length = 5
start_time = datetime.strptime("2019022701", "%Y%m%d%H")
end_time = datetime.strptime("2019030123", "%Y%m%d%H")
all_hours = pd.date_range(start=start_time, end=end_time, freq='h')

DEM_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/DEM_Sonoma_mean.npy"
DEM_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/DEM_Sonoma_std_dev.npy"

DEM_mean = np.mean(np.load(DEM_mean_file))
DEM_std = np.mean(np.load(DEM_std_file))

flow_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/Streamflow_Sonoma_mean.npy"
flow_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/Streamflow_Sonoma_std_dev.npy"

flow_mean = np.mean(np.load(flow_mean_file))
flow_std = np.mean(np.load(flow_std_file))

H1_prec_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/H1_prec_mean.npy"
H1_prec_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/H1_prec_std_dev.npy"

H1_prec_mean = np.mean(np.load(H1_prec_mean_file))
H1_prec_std = np.mean(np.load(H1_prec_std_file))

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # --- Dataset creation ---
    # dataset = FloodDataset(dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, all_hours, sequence_length,
    #                 DEM_mean, DEM_std, H1_prec_mean, H1_prec_std,
    #                 flood_mean=None, flood_std=None, return_metadata=False, mode='train')
    # main()



if sys.argv[1] == "train":

    # Prepare full dataset once
    dataset = FloodDataset(dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, all_hours, sequence_length,
                        DEM_mean, DEM_std, H1_prec_mean, H1_prec_std,
                        flood_mean=None, flood_std=None, return_metadata=False, mode='train')
    # print(dataset.samples[0])
    
    from collections import defaultdict

    # Initialize dataset
    dataset = FloodDataset(dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, all_hours, sequence_length,
                        DEM_mean, DEM_std, H1_prec_mean, H1_prec_std,
                        flood_mean=None, flood_std=None, return_metadata=False, mode='train')

    # Group sample indices by date
    samples_by_date = defaultdict(list)
    for i, (_, _, target_hour) in enumerate(dataset.samples):
        date = target_hour.date()
        samples_by_date[date].append(i)
        
        
    # Test the first date and first sample
    test_date = list(samples_by_date.keys())[0]
    # print(f"Testing date: {test_date}")

    # sample_indices = samples_by_date[test_date]
    # print(f"Sample count: {len(sample_indices)}")

    # # Try loading the first sample
    # i = sample_indices[0]
    # print(f"Loading dataset[{i}]...")
    # sample = dataset[i]
    # print("Sample loaded successfully!")
    
    from torch.utils.data import Subset, DataLoader

    for date, indices in tqdm(samples_by_date.items(), desc="Saving per-day datasets"):
        print(f"Processing {date} with {len(indices)} samples")
        if os.path.exists(f"{model_dir}/flood_dataset_{date}.pt"):
            print(f"Dataset for {date} already exists, skipping...")
            continue        
        
        # Create a subset and loader with multiple workers
        daily_subset = Subset(dataset, indices)
        daily_loader = DataLoader(daily_subset, batch_size=32, num_workers=4)

        # Accumulate the loaded data
        daily_data = []
        for batch in tqdm(daily_loader, desc=f"Loading samples for {date}"):
            if isinstance(batch, list):
                daily_data.extend(batch)  # For return_metadata=True
            else:
                daily_data.extend(zip(*batch))  # For multiple outputs

        # Save
        torch.save(daily_data, f"{model_dir}/flood_dataset_{date}.pt")
        
        daily_dataset = StaticFloodDataset(daily_data)


        
        
        
    # if not os.path.exists(f"{model_dir}/flood_dataset_{date}.pt"):
    #     print("Dataset not found, preparing full dataset...")

    # # Save samples grouped by date
    #     for date, indices in tqdm(samples_by_date.items(), desc="Saving per-day datasets"):
    #         daily_data = [dataset[i] for i in indices]
    #         torch.save(daily_data, f"{model_dir}/flood_dataset_{date}.pt")
            
    #     for date, indices in tqdm(samples_by_date.items(), desc="Saving per-day datasets"):
    #         print(f"Processing date: {date} with {len(indices)} samples...")
    #         daily_data = [dataset[i] for i in tqdm(indices, desc=f"Generating samples for {date}")]
    #         torch.save(daily_data, f"{model_dir}/flood_dataset_{date}.pt")

    #     print(f"Saved {len(samples_by_date)} daily datasets in {model_dir}")
    
    
    
    
    # # all_data = [dataset[i] for i in tqdm(range(len(dataset)))]
    # # torch.save(all_data, f"{model_dir}/flood_dataset_all.pt")
    # # print(f"Full dataset saved to {model_dir}/flood_dataset_all.pt")
    
    # # print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
    # # print(torch.cuda.memory_reserved() / 1e9, "GB reserved")
    
    # else:
    #     print(f"Dataset already exists at {model_dir}/flood_dataset_{date}.pt), loading...")
    #     data = torch.load(f"{model_dir}/flood_dataset_{date}.pt")
    #     dataset = StaticFloodDataset(data)
        
    #     # Collect all saved .pt files by date
    #     pt_files = sorted(glob(os.path.join(model_dir, "flood_dataset_*.pt")))

    #     # Load and concatenate all data
    #     all_data = []
    #     for f in pt_files:
    #         data = torch.load(f)
    #         all_data.extend(data)

    #     # Create final dataset object
    #     dataset = StaticFloodDataset(all_data)
    #     print(f"Combined dataset with {len(dataset)} samples loaded.")
    
    


    
    

    # print(dataset[0])
    kfold_train(daily_dataset, k_folds=3, batch_size=32, model_dir=model_dir, num_epochs=2)
    
elif sys.argv[1] == "dataset":
       # Prepare full dataset once
    dataset = FloodDataset(dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, all_hours, sequence_length,
                           DEM_mean, DEM_std, H1_prec_mean, H1_prec_std,
                           flood_mean=None, flood_std=None, return_metadata=False, mode='train')
    all_dates = sorted(list({s[2].date() for s in dataset.samples}))
    all_data = [dataset[i] for i in tqdm(range(len(dataset)))]
    torch.save(all_data, f"{model_dir}/flood_dataset_all.pt")
    print(f"Full dataset saved to {model_dir}/flood_dataset_all.pt")
    
    print(f"Dataset already exists at {model_dir}/flood_dataset_all.pt, loading...")
    all_data = torch.load(f"{model_dir}/flood_dataset_all.pt")
    dataset = StaticFloodDataset(all_data)





