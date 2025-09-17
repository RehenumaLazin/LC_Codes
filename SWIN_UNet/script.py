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


from timm.models.swin_transformer import SwinTransformer
from einops import rearrange

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





class FloodDataset(Dataset):
    def __init__(self, event, dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, hours, seq_len,
                 DEM_mean, DEM_std, precip_mean, precip_std,
                 flood_mean=None, flood_std=None, output_geotiff_dir=None, return_metadata=False, mode='train'):
        self.event = event
        self.dem_dir = dem_dir
        self.lc_dir = lc_dir
        self.precip_dir = precip_dir
        self.flood_dir = flood_dir
        self.geo_dir = output_geotiff_dir
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



    def _generate_samples(self):
        samples = []
        for tile in self.tiles:
            # event = "event1"
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
        
        static_input = self._load_static_inputs(tile)

        # static_input = self.static_inputs[tile]  # Already (2, H, W)

        # Load precipitation sequence (T, H, W)
        precip_seq = []
        for h in seq_hours:
            p_path = os.path.join(self.precip_dir, f"1H_prec_{self.event}_{h.strftime('%Y%m%d%H')}_{tile}")

                # print((output_geotiff_path))
            # print(p_path, os.path.exists(p_path))
            p, _, _ = self._read_raster(p_path)
            p_norm = (p - self.precip_mean) / self.precip_std
            precip_seq.append(p_norm)
            
        if self.geo_dir is not None and self.mode == 'test':
            output_geotiff_path = os.path.join(self.geo_dir, f"{target_hour.strftime('%Y%m%d%H')}_{tile}")
        precip_seq = np.stack(precip_seq, axis=0).astype(np.float32)
        precip_tensor = torch.from_numpy(precip_seq)
        _, transform, crs = self._read_raster(p_path)
            
        
        if self.mode == 'train':

            # Load label
            label_path = os.path.join(self.flood_dir, f"{self.event}_{target_hour.strftime('%Y%m%d%H')}_{tile}")
            # print(label_path, os.path.exists(label_path))
            label, transform, crs = self._read_raster(label_path)

            if self.flood_mean is not None and self.flood_std is not None and self.mode == 'train':
                label = (label - self.flood_mean) / self.flood_std

            label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        

        if self.return_metadata:
            return static_input, precip_tensor, output_geotiff_path, transform, crs
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


# Simple Transformer Block for decoder attention
# --- Attention Block for Decoder ---
import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ffn(x))
        return x

# --- Patch Expand Block ---
class PatchExpand(nn.Module):
    def __init__(self, in_dim, out_dim=None, expand_ratio=2):
        super().__init__()
        self.expand_ratio = expand_ratio
        out_dim = out_dim or in_dim // (expand_ratio ** 2)
        self.linear = nn.Linear(in_dim, out_dim * (expand_ratio ** 2))
        self.pixel_shuffle = nn.PixelShuffle(expand_ratio)

    def forward(self, x, H, W):
        print(x.shape, H, W)
        B= x.shape[0]
        C = x.shape[-1]
        x = x.view(B, H * W, C)  # reshape to (B, N, C)
        x = self.linear(x)  # (B, N, out_dim * r^2)
        x = x.permute(0, 2, 1).contiguous().view(B, -1, int(H), int(W))  # (B, out_dim * r^2, H, W)
        x = self.pixel_shuffle(x)  # (B, out_dim, H*r, W*r)
        x = x.flatten(2).transpose(1, 2)  # (B, new_N, out_dim)
        return x

# --- Swin-UNet ---
class SwinUNet(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=8, num_classes=1):
        super().__init__()
        self.patch_size = patch_size

        self.encoder = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
        )

        self.decode4 = PatchExpand(768, 384)
        self.attn4 = TransformerBlock(384)

        self.decode3 = PatchExpand(384, 192)
        self.attn3 = TransformerBlock(192)

        self.decode2 = PatchExpand(192, 96)
        self.attn2 = TransformerBlock(96)

        self.decode1 = PatchExpand(96, 48)
        self.attn1 = TransformerBlock(48)

        self.final_conv = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, static_input, precip_tensor):
        
        x = torch.cat([static_input, precip_tensor], dim=1)  # (B, 8, H, W)
        B = x.shape[0]
        H = W = x.size(2) // self.patch_size

        # Correctly apply encoder
        x = self.encoder.forward_features(x)  # returns [B, N, C]
        # x = torch.cat([static_input, precip_tensor], dim=1)  # (B, 8, H, W)
        # B = x.size(0)
        # H = W = x.size(2) // self.patch_size

        # x = self.encoder.patch_embed(x)
        # x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # # Corrected loop: process through BasicLayer
        # for blk in self.encoder.layers:
        #     x = blk(x)

        # x = self.encoder.norm(x)



        # Decoder with attention
        x = self.decode4(x, H // 8, W // 8)
        x = self.attn4(x)

        x = self.decode3(x, H // 4, W // 4)
        x = self.attn3(x)

        x = self.decode2(x, H // 2, W // 2)
        x = self.attn2(x)

        x = self.decode1(x, H, W)
        x = self.attn1(x)

        x = x.permute(0, 2, 1).contiguous().view(B, 48, H * 2, W * 2)
        x = self.final_conv(x)  # (B, 1, H, W)
        return x




def train(model, train_loader, val_loader, optimizer, criterion, checkpoint_path, scaler, fold_idx, start_epoch, num_epochs=10):
    model.train()
    train_losses, val_losses, mse = [], [], []

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        torch.cuda.empty_cache()

        for static, seq, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()

            # Do NOT move to specific device manually — let DataParallel handle this
            with autocast():
                output = model(static, seq)
                loss = criterion(output, target.to(output.device))  # only move target to match output

            if loss is not None and torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * static.size(0)
            else:
                print(f"[WARNING] Skipping invalid loss at epoch {epoch}")

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"[Train] Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, fold_idx, checkpoint_path)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        mse_score = 0.0
        with torch.no_grad():
            for static, seq, target in val_loader:
                with autocast():
                    output = model(static, seq)
                    loss = criterion(output, target.to(output.device))
                    val_loss += loss.item() * static.size(0)
                    mse_score += MSE(output, target.to(output.device))

        val_loss /= len(val_loader.dataset)
        mse_score /= len(val_loader)
        val_losses.append(val_loss)
        mse.append(mse_score)
        print(f"[Val] Epoch {epoch+1}: Loss: {val_loss:.4f}, MSE: {mse_score:.4f}")
        save_checkpoint(model, optimizer, epoch, fold_idx, checkpoint_path)
        
        # Switch model back to training mode after validation
        model.train()

    return model, train_losses, val_losses, mse


# --- Testing ---
def test(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, target in tqdm(dataloader, desc="Testing"):
            static = inputs[:, :2] 
            seq = inputs[:, 2:].unsqueeze(2) 
            # static, seq, target = static.to(device), seq.to(device), target.to(device)
            output = model(static, seq)
            # loss = criterion(output, target)
            loss = criterion(output, target.to(output.device))  # match output device
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")
    
    
# Load the saved model for testing
# def load_model(model_path, device='cuda'):
#     model = UNetLSTM()
#     model = nn.DataParallel(model).to(device)
#     checkpoint = torch.load(model_path, map_location=device)

#     if 'model_state_dict' in checkpoint:
#         # Full checkpoint (from save_checkpoint)
#         model.module.load_state_dict(checkpoint['model_state_dict'])
#     else:
#         # Just a raw state_dict (no wrapper)
#         model.module.load_state_dict(checkpoint)
#     return model




def load_model(model_path, device='cuda'):
    model = SwinUNet()
    # model = model.to(device)  # Move to device first
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Wrap in DataParallel *after* moving to device
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    # Load correct state_dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Strip "module." prefix if model is not wrapped in DataParallel
    if not isinstance(model, nn.DataParallel) and any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    return model,optimizer




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
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()  # unwrap
    else:
        model_state_dict = model.state_dict()
    checkpoint = {
        'epoch': epoch,
        'fold': fold,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at fold {fold}, epoch {epoch} to {checkpoint_path}")




def load_checkpoint(checkpoint_path, model, optimizer, device='cuda'):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = model.to(device)
        # if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        #     model = nn.DataParallel(model)

        state_dict = checkpoint['model_state_dict']
        # print("state_dict", state_dict.keys())
        # if not isinstance(model, nn.DataParallel) and any(k.startswith("module.") for k in state_dict.keys()):
        #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
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
    if len(batch[0]) == 3:  # Training mode
        static_inputs = torch.stack([item[0] for item in batch])       # (B, 2, H, W)
        precip_tensors = torch.stack([item[1] for item in batch])      # (B, T, H, W)
        label_tensors = torch.stack([item[2] for item in batch])       # (B, 1, H, W)
        return static_inputs, precip_tensors, label_tensors
    elif len(batch[0]) == 5:  # Testing mode
        # inputs = torch.stack([item[0] for item in batch])  # Stack inputs
        # transforms = [item[1] for item in batch]  # Extract transforms
        # crs_list = [item[2] for item in batch]  # Extract CRS
        # return inputs, transforms, crs_list
        
        static_inputs = torch.stack([item[0] for item in batch])       # (B, 2, H, W)
        precip_tensors = torch.stack([item[1] for item in batch])      # (B, T, H, W)
        paths = [item[2] for item in batch]    # (B, 1, H, W)
        transforms = [item[3] for item in batch]                       # List of rasterio transforms
        crs_list = [item[4] for item in batch]                         # List of CRS objects

        return static_inputs, precip_tensors, paths, transforms, crs_list

def MSE(pred, target): #(label - np.mean(FD_mean)) / np.mean(FD_std)
    mse = F.mse_loss(pred, target, reduction='mean').item()
    return mse

def NSE(pred, target, eps=1e-6):
    # pred and target are tensors
    # eps avoids division by zero

    # 1. Calculate numerator: sum of squared errors
    numerator = torch.sum((target - pred) ** 2)

    # 2. Calculate denominator: total variance of target
    denominator = torch.sum((target - torch.mean(target)) ** 2) + eps

    # 3. Calculate NSE
    nse = 1 - (numerator / denominator)

    return nse.item()
    
def kfold_train(dataset, k_folds, prev_trained_days, batch_size, model_dir, num_epochs):
    model = SwinUNet()
    # model = nn.DataParallel(model)
    model = model.to(torch.device("cuda"))
    
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        print(f"Model is on: {next(model.parameters()).device}")
    checkpoint_path = os.path.join(model_dir, "model_checkpoint.pth")
    
    # Extract all unique days (dates only)
    all_dates = sorted(list({s[2].date() for s in dataset.samples}))
    print(all_dates)
    # print("Sample entry example:", dataset.samples[0])
    # print("Type of s[2]:", type(dataset.samples[0][2]))
    # print(dataset.samples)
    # print(len(all_dates),all_dates,  {s[2].date() for s in dataset.samples} )
    assert len(all_dates) >= k_folds - prev_trained_days, "Not enough unique days for the requested number of folds."
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    

    criterion = nn.MSELoss()
    
    
    start_epoch, start_fold = load_checkpoint(checkpoint_path, model, optimizer)

    fold_losses = []

    for fold_idx in range(start_fold, k_folds):
        val_day = all_dates[fold_idx - prev_trained_days]
        print('val_day', val_day)

        train_indices = [i for i, s in enumerate(dataset.samples) if s[2].date() != val_day]
        val_indices = [i for i, s in enumerate(dataset.samples) if s[2].date() == val_day]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, collate_fn=custom_collate_fn, pin_memory=True)
        # print("train_loader", len(train_loader.dataset), len(train_loader.dataset[0]))
        # for i, sample in enumerate(train_loader):
        #     if sample is None:
        #         print(f"None returned at batch {i}")
        #         continue
        #     print(f"Batch {i} loaded:", [x.shape for x in sample])
        #     break
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, collate_fn=custom_collate_fn, pin_memory=True)

        model = SwinUNet()
        model = nn.DataParallel(model)
        model = model.to(torch.device("cuda"))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = GradScaler()
        criterion = nn.MSELoss()
        
        model_save_path = os.path.join(model_dir, f"unet_model_fold_{fold_idx}.pt") #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/unet_model_fold_{fold + 1}.pt"

        if fold_idx > 0 and os.path.exists(model_save_path):
            
            model, optimizer = load_model(model_save_path)
            model = nn.DataParallel(model).to(device)

        print(f"Starting Fold {fold_idx+1}/{k_folds}, Validation Day: {val_day}")
        scaler = GradScaler()

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

def test_unet_on_dataset(model, dataset, device):
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

    # Move and wrap model once
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    print(f"Model is on: {next(model.parameters()).device}")

    model.eval()
    with torch.no_grad():
        for static_input, precip_tensor, output_geotiff_paths, transforms, crs_list in tqdm(dataloader, desc="Testing"):
            static_input = static_input.to(device)
            precip_tensor = precip_tensor.to(device)

            outputs = model(static_input, precip_tensor)  # (B, 1, H, W)
            if outputs.dim() == 4:
                outputs = outputs.squeeze(1)  # (B, H, W)
            outputs = outputs.cpu().numpy()
            # print(output_geotiff_paths)
            

            for i in range(len(output_geotiff_paths)):
                out_arr = outputs[i].astype(np.float32)

                with rasterio.open(
                    output_geotiff_paths[i],
                    'w',
                    driver='GTiff',
                    height=out_arr.shape[0],
                    width=out_arr.shape[1],
                    count=1,
                    dtype='float32',
                    crs=crs_list[i],
                    transform=transforms[i]
                ) as dst:
                    dst.write(out_arr, 1)


    

# event_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/event2"
events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_DEM_tiles.csv" #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma_hourly_event2.csv"#f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma_hourly_event2_test.csv"

tile_csv = events_file


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


model_dir = f"/p/vast1/lazin1/UNet_trains/Sonoma_event2_SWIN_UNet"
os.makedirs(model_dir, exist_ok=True)

if sys.argv[1] == "train":
    
    
    event = "event3"
    

    
    
    dem_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma"
    lc_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_LC/Sonoma"
    # flood_paths.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/{event}/{tile_name}")

    precip_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/1H_prec/Sonoma/{event}"
    flood_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/{event}"

    # --- Time setup ---
    sequence_length = 6
    start_time = datetime.strptime("2017010901", "%Y%m%d%H")
    end_time = datetime.strptime("2017011323", "%Y%m%d%H")
    all_hours = pd.date_range(start=start_time, end=end_time, freq='h')
    dataset = FloodDataset(event, dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, all_hours, sequence_length,
                            DEM_mean, DEM_std, H1_prec_mean, H1_prec_std,
                            flood_mean=None, flood_std=None, output_geotiff_dir=None, return_metadata=False, mode='train')

    kfold_train(dataset, k_folds=5, batch_size=4, prev_trained_days=0,  model_dir=model_dir, num_epochs=10)
    

    
    
elif sys.argv[1] == "test":
    event = "event2"
    # Load the trained model
    model_path = f"{model_dir}/unet_model_fold_11.pt"
    model, _ = load_model(model_path, device=device)
    output_geotiff_dir =f"/p/vast1/lazin1/UNet_Geotiff_output/Sonoma_event2_prec_dem_LC_Hourly_script_fold11"
    os.makedirs(output_geotiff_dir, exist_ok=True)
    
    
    dem_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma"
    lc_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_LC/Sonoma"
    # flood_paths.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/{event}/{tile_name}")

    precip_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/1H_prec/Sonoma/{event}"
    flood_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/{event}"
    
    
    
    sequence_length = 5
    start_time = datetime.strptime("2019022500", "%Y%m%d%H")
    end_time = datetime.strptime("2019030123", "%Y%m%d%H")
    all_hours = pd.date_range(start=start_time, end=end_time, freq='h')
    dataset = FloodDataset(event, dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, all_hours, sequence_length,
                        DEM_mean, DEM_std, H1_prec_mean, H1_prec_std,
                        flood_mean=None, flood_std=None, output_geotiff_dir=output_geotiff_dir, return_metadata=True, mode='test')
    
    test_unet_on_dataset(model, dataset, device)
    





