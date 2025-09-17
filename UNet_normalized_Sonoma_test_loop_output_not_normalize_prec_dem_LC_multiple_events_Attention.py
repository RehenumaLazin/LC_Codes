import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import os
import json
import sys
import pandas as pd
import pickle
import csv
from torch.utils.data.sampler import SubsetRandomSampler
import rasterio
import torch.nn.functional as F
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm



device_ids = [1, 2,3] # List of GPU IDs to use
# import rasterio
# from rasterio.transform import from_origin
# from rasterio import warp


# Define Lazy Loading Dataset
# class FloodDataset(Dataset):
#     def __init__(self, input_geotiff_paths, label_geotiff_path=None,return_metadata=False):
#         """
#         Args:
#             input_geotiff_paths (list of str): List of 5 paths to input GeoTIFF files.
#             label_geotiff_path (str): Path to the label GeoTIFF file.
#         """
#         self.input_geotiff_paths = input_geotiff_paths
#         self.label_geotiff_path = label_geotiff_path
#         self.return_metadata = return_metadata

#     def __len__(self):
#         # Assuming all input GeoTIFFs have the same number of files
#         return len(self.input_geotiff_paths[0])

#     def __getitem__(self, idx):
#         # Read 6 GeoTIFF inputs
#         inputs = []
#         for path in self.input_geotiff_paths:
#             with rasterio.open(path[idx]) as src:
#                 inputs.append(src.read(1))  # Read the first band
#                 transform = src.transform
#                 crs = src.crs

#         # Stack the inputs along the channel dimension
#         inputs = np.stack(inputs, axis=0)  # Shape: (5, H, W)
        
#         # Normalize inputs and labels
#         # inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
#         input_array_normalized = np.empty(inputs.shape, dtype=np.float32)
#         input_array_normalized[0,:,:] = (inputs[0,:,:] - np.mean(DEM_mean)) / np.mean(DEM_std)
#         # input_array_normalized[1,:,:] =  np.where(inputs[1,:,:] == 11, 1, 0)     
#         input_array_normalized[1,:,:] = (inputs[1,:,:] - np.mean(day1_prec_mean)) / np.mean(day1_prec_std)
#         input_array_normalized[2,:,:] = (inputs[2,:,:] - np.mean(day5_prec_mean)) / np.mean(day5_prec_std)
#         # input_array_normalized[4,:,:] = (inputs[4,:,:] - np.mean(day30_prec_mean)) / np.mean(day30_prec_std)
#         # input_array_normalized[4,:,:] = (inputs[4,:,:] - np.mean(flow_mean)) / np.mean(flow_std)
#         # array = inputs[5,:,:]
#         # array[array == -9999] = 1  # Permanent waterbody is -9999
#         # input_array_normalized[5,:,:] = (array - np.mean(SM_mean)) / np.mean(SM_std)
        
#         inputs = torch.tensor(input_array_normalized, dtype=torch.float32)
#         # print(inputs.shape, 'inputs', self.label_geotiff_path, self.return_metadata)
#         if self.label_geotiff_path:
            
#             # Read label GeoTIFF
#             with rasterio.open(self.label_geotiff_path[idx]) as src:
#                 label = src.read(1)  # Shape: (H, W)

#             # label = (label > 0).astype(np.float32)
#             # label = (label - np.mean(FD_mean)) / np.mean(FD_std)
#             label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
#             if self.return_metadata:
#                 return inputs, label, transform, crs
#             return inputs, label
            
#         if self.return_metadata:
#             # print(inputs, transform, crs)
#             return inputs, transform, crs

#         return inputs           
#         # print(inputs, transform, crs)
#         # return inputs, transform, crs

class FloodDataset(Dataset):
    def __init__(self, event, dem_dir, lc_dir, D1_prec_dir, D5_prec_dir, flood_dir, tile_csv, all_days,
                            DEM_mean, DEM_std, day1_prec_mean, day1_prec_std, day5_prec_mean, day5_prec_std,
                 flood_mean=None, flood_std=None, output_geotiff_dir=None, return_metadata=False, mode='train'):
        self.event = event
        self.dem_dir = dem_dir
        self.lc_dir = lc_dir
        self.D1_prec_dir = D1_prec_dir
        self.D5_prec_dir = D5_prec_dir
        self.flood_dir = flood_dir
        self.geo_dir = output_geotiff_dir
        self.days = all_days
        self.return_metadata = return_metadata
        self.mode = mode.lower()

        self.DEM_mean = DEM_mean
        self.DEM_std = DEM_std
        self.day1_prec_mean = day1_prec_mean
        self.day1_prec_std = day1_prec_std
        self.day5_prec_mean = day5_prec_mean
        self.day5_prec_std = day5_prec_std
        self.flood_mean = flood_mean
        self.flood_std = flood_std

        self.tiles = pd.read_csv(tile_csv)['ras_name'].tolist()
        
        self.samples = self._generate_samples()



    def _generate_samples(self):
        samples = []
        for tile in self.tiles:
            # event = "event1"
            for i in range(len(self.days)):
                # seq_hours = self.hours[i - self.seq_len + 1:i + 1]
                target_day = self.days[i]
                # if all(os.path.exists(os.path.join(self.precip_dir, f"1H_prec_{event}_{h.strftime('%Y%m%d%H')}_{tile}")) for h in seq_hours) and \
                #    os.path.exists(os.path.join(self.flood_dir, f"{event}_{target_hour.strftime('%Y%m%d%H')}_{tile}")):
                #     samples.append((tile, seq_hours, target_hour))
                samples.append((tile, target_day))
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
        # static_input = np.stack([dem_norm,], axis=0)
        # return dem_norm
        return torch.from_numpy(static_input)
        # return torch.from_numpy(static_input)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tile, target_day = self.samples[idx]
        
        static_inputs = self._load_static_inputs(tile)

        # static_input = self.static_inputs[tile]  # Already (2, H, W)

        # Load precipitation sequence (T, H, W)
        # precip_arr = []
        # for h in seq_hours:
        
        D1_p_path = os.path.join(self.D1_prec_dir, f"1day_prec_{self.event}_{target_day.strftime('%Y-%m-%d')}_{tile}") #1day_prec_event3_2017-02-04_Sonoma_crop_9.tif
        D5_p_path = os.path.join(self.D5_prec_dir, f"5day_prec_{self.event}_{target_day.strftime('%Y-%m-%d')}_{tile}") 

            # print((output_geotiff_path))
        # print(p_path, os.path.exists(p_path))
        p_1d, _, _ = self._read_raster(D1_p_path)
        p_5d, _, _ = self._read_raster(D5_p_path)
        p1d_norm = (p_1d - self.day1_prec_mean) / self.day1_prec_std
        p5d_norm = (p_5d - self.day5_prec_mean) / self.day5_prec_std
        
        precs = np.stack([p1d_norm, p5d_norm], axis=0)
        # print('PRINT',static_inputs.shape, torch.from_numpy(p1d_norm).unsqueeze(0).shape, torch.from_numpy(p5d_norm).unsqueeze(0).shape)
        
        inputs = np.concatenate([static_inputs.numpy(), precs], axis=0).astype(np.float32)
        # input_tensor = torch.from_numpy(inputs)  # shape: (4, H, W)
        # inputs = np.stack([static_inputs, torch.from_numpy(precs)], axis=0).astype(np.float32)
        
        

            
        if self.geo_dir is not None and self.mode == 'test':
            output_geotiff_path = os.path.join(self.geo_dir, f"{target_day.strftime('%Y-%m-%d')}_{tile}")
        # precip_seq = np.stack(precip_seq, axis=0).astype(np.float32)
        input_tensor = torch.from_numpy(inputs)
        _, transform, crs = self._read_raster(D1_p_path)
            
        
        if self.mode == 'train':

            # Load label
            label_path = os.path.join(self.flood_dir, f"{self.event}_{target_day.strftime('%Y-%m-%d')}_{tile}")
            # print(label_path, os.path.exists(label_path))
            label, transform, crs = self._read_raster(label_path)

            if self.flood_mean is not None and self.flood_std is not None and self.mode == 'train':
                label = (label - self.flood_mean) / self.flood_std

            label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        

        if self.return_metadata:
            return input_tensor, output_geotiff_path, transform, crs
        return input_tensor, label_tensor


# import rasterio
# from rasterio.transform import Affine

# GPU selection
# if cfg["gpu"] >= 0:
#     device = f"cuda:{cfg['gpu']}"
# else:
#     device = 'cpu'

# global DEVICE
# DEVICE = torch.device("gpu" if torch.cuda.is_available() else "cpu")
# U-Net Model Definition with initial output channels set to 32
# class UNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1, init_features=32):
#         super(UNet, self).__init__()
        
#         features = init_features

#         def conv_block(in_channels, out_channels):
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True)
#             )

#         self.encoder1 = conv_block(in_channels, features)
#         self.encoder2 = conv_block(features, features * 2)
#         self.encoder3 = conv_block(features * 2, features * 4)
#         self.encoder4 = conv_block(features * 4, features * 8)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.middle = conv_block(features * 8, features * 16)

#         self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
#         self.decoder4 = conv_block(features * 16, features * 8)
#         self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
#         self.decoder3 = conv_block(features * 8, features * 4)
#         self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
#         self.decoder2 = conv_block(features * 4, features * 2)
#         self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
#         self.decoder1 = conv_block(features * 2, features)

#         self.out_conv = nn.Conv2d(features, out_channels, kernel_size=1)

#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool(enc1))
#         enc3 = self.encoder3(self.pool(enc2))
#         enc4 = self.encoder4(self.pool(enc3))

#         mid = self.middle(self.pool(enc4))

#         dec4 = self.upconv4(mid)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)

#         return self.out_conv(dec1)

import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    """
    g: decoder feature (gating signal)
    x: encoder feature (skip connection)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # align channel dims to an intermediate space, add, activate, then gate x
        att = self.relu(self.W_g(g) + self.W_x(x))
        att = self.psi(att)              # [B,1,H,W] weights in [0,1]
        return x * att                   # gate the skip features


class UNet_Attention(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, init_features=32):
        super().__init__()
        features = init_features

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.encoder1 = conv_block(in_channels, features)            # -> F
        self.encoder2 = conv_block(features, features * 2)           # -> 2F
        self.encoder3 = conv_block(features * 2, features * 4)       # -> 4F
        self.encoder4 = conv_block(features * 4, features * 8)       # -> 8F
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.middle = conv_block(features * 8, features * 16)        # -> 16F

        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, 2)
        self.decoder4 = conv_block(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, 2)
        self.decoder3 = conv_block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.decoder2 = conv_block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.decoder1 = conv_block(features * 2, features)

        self.out_conv = nn.Conv2d(features, out_channels, kernel_size=1)

        # Attention gates for each skip level
        self.att4 = AttentionGate(F_g=features * 8, F_l=features * 8, F_int=features * 4)
        self.att3 = AttentionGate(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.att2 = AttentionGate(F_g=features * 2, F_l=features * 2, F_int=features)
        self.att1 = AttentionGate(F_g=features,       F_l=features,       F_int=max(1, features // 2))

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)                    # F
        enc2 = self.encoder2(self.pool(enc1))      # 2F
        enc3 = self.encoder3(self.pool(enc2))      # 4F
        enc4 = self.encoder4(self.pool(enc3))      # 8F

        # Bottleneck
        mid  = self.middle(self.pool(enc4))        # 16F

        # Decoder with attention-gated skips
        dec4 = self.upconv4(mid)                   # 8F
        enc4_att = self.att4(dec4, enc4)           # gate skip from enc4
        dec4 = self.decoder4(torch.cat([dec4, enc4_att], dim=1))  # -> 8F

        dec3 = self.upconv3(dec4)                  # 4F
        enc3_att = self.att3(dec3, enc3)
        dec3 = self.decoder3(torch.cat([dec3, enc3_att], dim=1))  # -> 4F

        dec2 = self.upconv2(dec3)                  # 2F
        enc2_att = self.att2(dec2, enc2)
        dec2 = self.decoder2(torch.cat([dec2, enc2_att], dim=1))  # -> 2F

        dec1 = self.upconv1(dec2)                  # F
        enc1_att = self.att1(dec1, enc1)
        dec1 = self.decoder1(torch.cat([dec1, enc1_att], dim=1))  # -> F

        return self.out_conv(dec1)


# Calculate mIoU
def mean_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).clamp(0, 1).sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def MSE(pred, target): #(label - np.mean(FD_mean)) / np.mean(FD_std)
    # std = np.mean(FD_std)
    # mean = np.mean(FD_mean)
    # pred_denorm = pred * std + mean
    # target_denorm = target * std + mean

    # ---- 2. Calculate MSE ----
    # mse = F.mse_loss(pred_denorm, target_denorm, reduction='mean').item()
    mse = F.mse_loss(pred, target, reduction='mean').item()

    
    return mse



# Load the saved model for testing
def load_model(model_path, device='cuda'):
    model = UNet_Attention(in_channels=4, out_channels=1, init_features=32)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    return model


import torch
import os

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

# def custom_collate_fn(batch):
#     # print(len(batch[0]))
#     """
#     Custom collate function for training and testing.

#     For training:
#         Returns only inputs and labels.

#     For testing:
#         Returns inputs, transforms, and CRS metadata.
#     """
#     if len(batch[0]) == 2:  # Training mode
#         inputs = torch.stack([item[0] for item in batch])  # Stack inputs
#         labels = torch.stack([item[1] for item in batch])  # Stack labels
#         return inputs, labels
#     elif len(batch[0]) == 3:  # Testing mode
#         inputs = torch.stack([item[0] for item in batch])  # Stack inputs
#         transforms = [item[1] for item in batch]  # Extract transforms
#         crs_list = [item[2] for item in batch]  # Extract CRS
#         return inputs, transforms, crs_list

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
        input_tensors = torch.stack([item[0] for item in batch])
        # static_inputs = torch.stack([item[0] for item in batch])       # (B, 2, H, W)
        # precip_tensors = torch.stack([item[1] for item in batch])      # (B, T, H, W)
        label_tensors = torch.stack([item[1] for item in batch])       # (B, 1, H, W)
        # return static_inputs, precip_tensors, label_tensors
        return input_tensors, label_tensors
    elif len(batch[0]) == 4:  # Testing mode
        # inputs = torch.stack([item[0] for item in batch])  # Stack inputs
        # transforms = [item[1] for item in batch]  # Extract transforms
        # crs_list = [item[2] for item in batch]  # Extract CRS
        # return inputs, transforms, crs_list
        
        # static_inputs = torch.stack([item[0] for item in batch])       # (B, 2, H, W)
        # precip_tensors = torch.stack([item[1] for item in batch])      # (B, T, H, W)
        input_tensors = torch.stack([item[0] for item in batch])
        paths = [item[1] for item in batch]    # (B, 1, H, W)
        transforms = [item[2] for item in batch]                       # List of rasterio transforms
        crs_list = [item[3] for item in batch]                         # List of CRS objects

        return input_tensors, paths, transforms, crs_list

def load_dict(file_path):
    """
    Loads a dictionary from a file using pickle.
    
    Args:
        file_path (str): Path to the file where the dictionary is saved.
    
    Returns:
        dict: The loaded dictionary.
    """
    with open(file_path, 'rb') as f:
        dictionary = pickle.load(f)
    print(f"Dictionary loaded from {file_path}")
    return dictionary

def load_unet_model(model_class, model_path, in_channels=4, out_channels=1, initial_filters=32, device=None):
    """
    Loads a U-Net model with specified parameters and handles possible missing/extra keys.

    Args:
        model_class (nn.Module): The class of the U-Net model to instantiate.
        model_path (str): Path to the saved model file (.pth or .pt).
        in_channels (int): Number of input channels. Default is 6.
        out_channels (int): Number of output channels. Default is 1.
        initial_filters (int): Number of filters in the first layer. Default is 32.
        device (torch.device, optional): Device to load the model on. If None, will use CUDA if available.

    Returns:
        nn.Module: The loaded U-Net model.
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model
    model = model_class(in_channels=in_channels, out_channels=out_channels, initial_filters=initial_filters)
    
    # Load the model's state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle DataParallel models by removing 'module.' prefix if necessary
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Load the state dict into the model with diagnostics for mismatched keys
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print("Missing keys in state_dict:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys in state_dict:", unexpected_keys)
    
    # Move model to the correct device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Model loaded successfully from {model_path}")
    return model




# Training function with AMP
def train_model(model, prev_trained_days, criterion, optimizer, train_loader, val_loader, num_epochs=int, start_epoch=int, fold=int, checkpoint_path=str, device='cuda', scaler=None):
    model.to(device)
    if scaler is None:
        scaler = GradScaler()
    # train_losses, val_losses, miou = [], [], []
    train_losses, val_losses, mse = [], [], []
    # start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    if start_epoch >= num_epochs:
        start_epoch = 0
    # print(start_epoch)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):#train_loader:
            # inputs = torch.cat((dem, precs), axis=0)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass with autocasting
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Backward pass with scaled gradients
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # running_loss += loss.item() * inputs.size(0)
            
            if loss is not None and torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * inputs.size(0)
            else:
                print(f"Skipping step due to invalid loss at epoch {epoch}")

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        # iou_score = 0.0
        mse_score = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                # inputs = torch.cat((dem, precs), axis=0)
                inputs, targets = inputs.to(device), targets.to(device)
                # print('inputs', inputs)
                
                with autocast():
                    outputs = model(inputs)
                    
                    # std = np.mean(FD_std)
                    # mean = np.mean(FD_mean)
                    # outputs_denorm = outputs * std + mean
                    # targets_denorm = targets * std + mean
                    # loss = criterion(outputs_denorm, targets_denorm)
                    loss = criterion(outputs, targets)
                    
                    
                    # print('output', outputs)
                    # print('target', targets)
                    # loss = criterion(outputs, targets)
                    
                val_loss += loss.item() * inputs.size(0)
                # iou_score += mean_iou(torch.sigmoid(outputs), targets)
                
                mse_score += MSE(outputs, targets)


        
        val_loss /= len(val_loader.dataset)
        # iou_score /= len(val_loader)
        mse_score /= len(val_loader)
        val_losses.append(val_loss)
        # miou.append(iou_score)
        mse.append(mse_score)
        # print(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {iou_score:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation MSE: {mse_score:.4f}")
        save_checkpoint(model, optimizer, epoch,fold + (prev_trained_days - 1), checkpoint_path)
        print(checkpoint_path, "saved")
        
    

    # return model, train_losses, val_losses, miou
    return model, train_losses, val_losses, mse
def k_fold_groups(dataset, k=5):
    kfold = KFold(n_splits=k, shuffle=True)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        all_indices = {'train_indices': [], 'val_indices': []}
        all_indices['train_indices'].append(train_idx.tolist())
        all_indices['val_indices'].append(val_idx.tolist())
                
        # Save indices to JSON
        with open(os.path.join(model_dir, str(fold+1) +"_train_val_indices.json"), "w") as f:
            json.dump(all_indices, f)
            

            
# K-Fold Cross-Validation
def k_fold_cross_validation_based_on_events(events, events_file, dataset, num_epochs=int,prev_trained_days=int, batch_size=128, model_dir = str, checkpoint_path=str, device='cuda', device_ids=device_ids):
    # input_dict = load_dict(path_input_dict)
    # target_dict = load_dict(path_target_dict)
    # events = pd.read_csv(event_file, header=None).to_numpy()
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        # model = torch.nn.DataParallel(model)
        
    
    model = UNet_Attention(in_channels=4, out_channels=1, init_features=32).to(device)
    model = nn.DataParallel(model).to(device)
    print(f"Model is on: {next(model.parameters()).device}")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    
    tile_names = pd.read_csv(events_file)['ras_name'].tolist() 
    # criterion = nn.BCEWithLogitsLoss() #nn.MSELoss())
    criterion = nn.MSELoss()
    
    
    start_epoch, start_fold = load_checkpoint(checkpoint_path, model, optimizer)
    if start_epoch == 50:
        start_epoch = 0
        
    if start_fold <0:
        start_fold = 0
        
    # start_epoch, start_fold = 0, 0
    # if start_epoch > num_epochs:
    #     start_epoch = 0
    #     start_fold += 1
    # print(start_epoch, start_fold)
    # all_indices = {'train_indices': [], 'val_indices': []}
    fold_losses = []
    # for fold in range( start_fold - (prev_trained_days - 1), len(events)):
    for fold in range( start_fold , len(events)):
        
        event = events[fold]
        print(fold, event, 'len(events)', len(events))
        
        model_save_path = os.path.join(model_dir, f"unet_model_fold_{fold + prev_trained_days }.pt") #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/unet_model_fold_{fold + 1}.pt"

        if (fold > 0 and os.path.exists(model_save_path)) or prev_trained_days>0:
            
            model = load_model(model_save_path)
            model = model.to(device)
            print(f"model {model_save_path} is loaded")
            # print(f"model {event} is already trained, moving to next event")
            
            # continue
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scaler = GradScaler()
            
        # elif fold==0:
        
        
        all_dates = sorted(list({s[1].date() for s in dataset.samples}))
        print(all_dates, len(all_dates))
        # print("Sample entry example:", dataset.samples[0])
        # print("Type of s[2]:", type(dataset.samples[0][2]))
        # print(dataset.samples)
        # print(len(all_dates),all_dates,  {s[2].date() for s in dataset.samples} )
        assert len(all_dates) >= len(events) , "Not enough unique days for the requested number of folds."
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = GradScaler()
    
    

        criterion = nn.MSELoss()
        
        
        # start_epoch, start_fold = load_checkpoint(checkpoint_path, model, optimizer)
        # if fold ==0:
        #     start_epoch = 0

        fold_losses = []

    # for fold_idx in range(start_fold, k_folds):
        val_day = all_dates[fold]
        print('val_day', val_day)

        train_indices = [i for i, s in enumerate(dataset.samples) if s[1].date() != val_day]
        val_indices = [i for i, s in enumerate(dataset.samples) if s[1].date() == val_day]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        
        

        # val_keys= [key for key in tile_names if event in key]
        # train_keys = [key for key in tile_names if event not in key]
        
        
        # train_idx = [
        #     idx for idx, tile_name in enumerate(tile_names)
        #     if any(keyword in tile_name for keyword in train_keys)
        # ]
        
        # val_idx = [
        #     idx for idx, tile_name in enumerate(tile_names)
        #     if any(keyword in tile_name for keyword in val_keys)
        # ]
        print(len(train_indices), len(val_indices), fold)
        # train_sampler = SubsetRandomSampler(train_idx)
        # val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate_fn)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=custom_collate_fn)
        
        # if len(train_loader) == 0:
        #     print("Warning: train_loader is empty. Skipping this fold.")
        #     continue

        
        
        print(len(train_indices), len(val_indices))
        print("START_TRAINING")
        print(num_epochs,start_epoch, fold, checkpoint_path)






        # optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # start_epoch, start_fold = load_checkpoint(checkpoint_path, model, optimizer)
        # if start_epoch > num_epochs:
        #     start_epoch = 0
        # print(start_epoch)

        # trained_model, train_losses, val_losses, miou = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs,start_epoch, fold, checkpoint_path, device)
        trained_model, train_losses, val_losses, mse = train_model(model, prev_trained_days, criterion, optimizer, train_loader, val_loader, num_epochs,start_epoch, fold, checkpoint_path, device, scaler=scaler)
        print("END_TRAINING", train_losses, val_losses)
        # Save the model after each fold
        fold_losses.append((train_losses, val_losses))
        model_save_path = os.path.join(model_dir, f"unet_model_fold_{fold+1+ prev_trained_days}.pt")
        torch.save(trained_model.state_dict(), model_save_path) #torch.save(model.state_dict(), str(model_path))
        # np.save(f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/train_val_losses_fold_{fold + 1}.npy" , fold_losses)
        np.save(os.path.join(model_dir, f"train_val_losses_fold_{fold + 1 + prev_trained_days}.npy") , fold_losses)
        # np.save(f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/val_miou_{fold + 1}.npy" , miou)
        np.save(os.path.join(model_dir, f"val_mse_{fold + 1 + prev_trained_days}.npy"),mse)
        print(f"Model for fold {fold + 1 + prev_trained_days} saved as {model_save_path}.\n")
        
                # Plot losses for this fold
        # plt.figure()
        # plt.plot(range(0, num_epochs ), train_losses, label='Train Loss')
        # plt.plot(range(0, num_epochs ), val_losses, label='Validation Loss')
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title(f"Fold {fold + 1} Loss")
        # plt.legend()
        # plt.savefig(os.path.join(model_dir, f"train_val_losses_fold_{fold + 1}_loss_curve.png"))
        # plt.close()

def test_unet_on_dataset(model, dataset, device):
    """
    Test the U-Net model on an unseen dataset and save results as GeoTIFFs.

    Args:
        model: The trained U-Net model.
        dataset: Dataset instance (FloodDataset).
        output_geotiff_paths (list of str): List of paths to save the output GeoTIFF files.
        device (str): Device to perform the computation on ('cpu' or 'cuda').

    Returns:
        None
    """
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    # print(dataloader)
    for i, data in tqdm(enumerate(dataloader), desc="Testing"):#enumerate(dataloader):
        # print(data, 'data')
        inputs, transform, crs = data[0].to(device), data[2][0], data[3][0]
        output_geotiff_paths = data[1][0]
        with torch.no_grad():
            outputs = model(inputs)  # Add batch dimension
            # outputs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            outputs = outputs.squeeze().cpu().numpy()
            
            
            # std = np.mean(FD_std)
            # mean = np.mean(FD_mean)
            # outputs_denorm = outputs * std + mean

        # Binarize the output (optional)
        # output_binary = (outputs > 0.5).astype(np.float32)

        # Save the result as a GeoTIFF
        with rasterio.open(
            output_geotiff_paths,
            "w",
            driver="GTiff",
            height=outputs.shape[0],
            width=outputs.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(outputs, 1)

        print(f"Output saved as GeoTIFF: {output_geotiff_paths}")





events_file = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'
combined_df = pd.read_csv(events_file, header=None) 
events = [i.split("/")[-1][:-4] for i in combined_df[0]]
# print(events)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEM_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/DEM_Sonoma_mean.npy"
DEM_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/DEM_Sonoma_std_dev.npy"

DEM_mean = np.mean(np.load(DEM_mean_file))
DEM_std = np.mean(np.load(DEM_std_file))

flow_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/Streamflow_Sonoma_mean.npy"
flow_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/Streamflow_Sonoma_std_dev.npy"

flow_mean = np.mean(np.load(flow_mean_file))
flow_std = np.mean(np.load(flow_std_file))

day1_prec_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/day1_prec_Sonoma_mean.npy"
day1_prec_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/day1_prec_Sonoma_std_dev.npy"

day1_prec_mean = np.mean(np.load(day1_prec_mean_file))
day1_prec_std= np.mean(np.load(day1_prec_std_file))

day5_prec_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/day5_prec_Sonoma_mean.npy"
day5_prec_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/day5_prec_Sonoma_std_dev.npy"

day5_prec_mean = np.mean(np.load(day5_prec_mean_file))
day5_prec_std = np.mean(np.load(day5_prec_std_file))

# day30_prec_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/day30_prec_mean.npy"
# day30_prec_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/day30_prec_std_dev.npy" 


# day30_prec_mean = np.mean(np.load(day30_prec_mean_file))
# day30_prec_std = np.mean(np.load(day30_prec_std_file))


SM_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/SM_Sonoma_mean.npy"
SM_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/SM_Sonoma_std_dev.npy"
SM_mean = np.mean(np.load(SM_mean_file))
SM_std = np.mean(np.load(SM_std_file))



FD_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/Flood_depth_Sonoma_mean.npy"
FD_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/Flood_depth_Sonoma_std_dev.npy"
FD_mean = np.mean(np.load(SM_mean_file))
FD_std = np.mean(np.load(SM_std_file))


tile_csv = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_DEM_tiles.csv" #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma_hourly_event2.csv"#f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma_hourly_event2_test.csv"


# if sys.argv[1] == "train":
    
    
#     event = "event3"
    

#     model_dir = f"/p/vast1/lazin1/UNet_trains/Sonoma_event2_output_not_normalized_prec_dem_retrain"
#     os.makedirs(model_dir,exist_ok=True)
    
#     dem_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma"
#     lc_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_LC/Sonoma"
#     # flood_paths.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/{event}/{tile_name}")

#     D1_prec_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/1D_prec/Sonoma/{event}/"
#     D5_prec_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/5D_prec/Sonoma/{event}/"
#     flood_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma/{event}"

#     # --- Time setup ---
#     start_time = datetime.strptime("20221230", "%Y%m%d")
#     end_time = datetime.strptime("20230120", "%Y%m%d")
#     all_days = pd.date_range(start=start_time, end=end_time, freq='d')
#     dataset = FloodDataset(event, dem_dir, lc_dir, D1_prec_dir, D5_prec_dir, flood_dir, tile_csv, all_days,
#                             DEM_mean, DEM_std, day1_prec_mean, day1_prec_std, day5_prec_mean, day5_prec_std,
#                             flood_mean=None, flood_std=None, output_geotiff_dir=None, return_metadata=False, mode='train')
    
#     checkpoint_path = os.path.join(model_dir,"model_checkpoint.pth")
    
#     k_fold_cross_validation_based_on_events(events, events_file, dataset, num_epochs=50, batch_size=128, model_dir=model_dir, checkpoint_path=checkpoint_path, device=device, device_ids=device_ids)


#     kfold_train(dataset, k_folds=5, batch_size=4, prev_trained_days=0,  model_dir=model_dir, num_epochs=10)

if sys.argv[1] == "train":
    event = f"event2"
# k_fold_groups(dataset, k=5)
# for e, EVENT_STR in enumerate(EVENT_STRS):
    # event_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/EVENT_{EVENT_STR}.csv" 
    event_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma/{event}"
    
    start_time = datetime.strptime("20190226", "%Y%m%d")
    end_time = datetime.strptime("20190304", "%Y%m%d")
    all_hours = pd.date_range(start=start_time, end=end_time, freq='d')
    # events = [ event for  event in (os.listdir(event_dir)) if os.path.isdir(os.path.join(event_dir, event))]
    events = []
    for h in all_hours:
        
        events.append(h.strftime('%Y-%m-%d'))

    model_dir = f"/p/vast1/lazin1/UNet_trains/Sonoma_multiple_DEM_LC_prec_multiple_Attention"
    os.makedirs(model_dir,exist_ok=True)
    
    
    checkpoint_path = os.path.join(model_dir,"model_checkpoint.pth")

                
    # events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma_event2.csv" #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined_{EVENT_STR}_No_Threshold.csv" #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined.csv"  #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'
    events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_DEM_tiles.csv" #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma_hourly_event2.csv"#f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma_hourly_event2_test.csv"
    
    tiles_df = pd.read_csv(events_file)
    tile_names = tiles_df['ras_name'].tolist()  # Assuming the column is named 'tile_name'


    # DEM = []
    # LC = []
    # day1_prec = []
    # day5_prec = []
    # streamflow = []
    # SM = []
    # label_geotiff_path = []
    # events = []
    # for tile_name in tile_names:

        
    #     DEM.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma/{tile_name}")
    #     for h in all_hours:
            
    #         day1_prec.append( f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/1D_prec/Sonoma/{event}/1day_prec_{event}_{h.strftime('%Y-%m-%d')}_{tile_name}")
    #         day5_prec.append( f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/5D_prec/Sonoma/{event}/5day_prec_{event}_{h.strftime('%Y-%m-%d')}_{tile_name}")
        
    #         label_geotiff_path.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma/{event}_{h.strftime('%Y-%m-%d')}_{tile_name}")
        
    #         events.append(h.strftime('%Y-%m-%d'))
    # # GeoTIFF paths

    #     input_geotiff_paths =[
    #         DEM,
    #         # LC,
    #         day1_prec,
    #         day5_prec,
    #         # streamflow,
    #         # SM
    #                         ]



    # # Dataset
    # dataset = FloodDataset(input_geotiff_paths, label_geotiff_path)
    
    
    dem_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma"
    lc_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_LC/Sonoma"
    # flood_paths.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/{event}/{tile_name}")

    D1_prec_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/1D_prec/Sonoma/{event}/"
    D5_prec_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/5D_prec/Sonoma/{event}/"
    flood_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma/{event}"

    # --- Time setup ---
    # start_time = datetime.strptime("20221230", "%Y%m%d")
    # end_time = datetime.strptime("20230120", "%Y%m%d")
    all_days = pd.date_range(start=start_time, end=end_time, freq='d')
    dataset = FloodDataset(event, dem_dir, lc_dir, D1_prec_dir, D5_prec_dir, flood_dir, tile_csv, all_days,
                            DEM_mean, DEM_std, day1_prec_mean, day1_prec_std, day5_prec_mean, day5_prec_std,
                            flood_mean=None, flood_std=None, output_geotiff_dir=None, return_metadata=False, mode='train')

    k_fold_cross_validation_based_on_events(events, events_file, dataset, prev_trained_days=0, num_epochs=50, batch_size=128, model_dir=model_dir, checkpoint_path=checkpoint_path, device=device, device_ids=device_ids)
            




elif sys.argv[1] == "test":
    events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma_event2.csv"#f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined_reduced_events_test.csv" #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined.csv"  #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'

    tiles_df = pd.read_csv(events_file)
    tile_names = tiles_df['ras_name'].tolist()  # Assuming the column is named 'tile_name'


    DEM = []
    LC = []
    day1_prec = []
    day5_prec = []
    streamflow = []
    SM = []
    label_geotiff_path = []
    # output_geotiff_paths =[]
    output_geotiff_dir =f"/p/vast1/lazin1/UNet_Geotiff_output/Sonoma_multiple_DEM_LC_prec_multiple_Attention/0912"
    os.makedirs(output_geotiff_dir, exist_ok=True)
    for tile_name in tile_names:
        tile_str = tile_name.split("Sonoma")[-1]#tile_name.split("crop")[0][:-1] 
        event = tile_name.split("/")[-1].split("_")[0]
        date = tile_name.split("/")[-1].split("_")[1]
        
        DEM.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma/Sonoma{tile_str}")
        LC.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_LC/Sonoma/LC_Sonoma{tile_str}")
        day1_prec.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/1D_prec/Sonoma/{event}/1day_prec_{tile_name}")
        day5_prec.append( f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/5D_prec/Sonoma/{event}/5day_prec_{tile_name}")
        streamflow.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/streamflow/Sonoma/{event}/streamflow_{tile_name}")
        SM.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_SM/Sonoma/{event}/SM_{tile_name}")
        
        label_geotiff_path.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma/{event}/{tile_name}") 
        # output_geotiff_paths.append(f"{output_geotiff_dir}/{tile_name}")
        
        
    # GeoTIFF paths

    input_geotiff_paths =[
        DEM,
        LC,
        day1_prec,
        day5_prec,
        # streamflow,
        # SM
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
    # dataset = FloodDataset(input_geotiff_paths, return_metadata=True)
    # data = dataset[0]
    
    dem_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma"
    lc_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_LC/Sonoma"
    # flood_paths.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/{event}/{tile_name}")

    D1_prec_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/1D_prec/Sonoma/{event}/"
    D5_prec_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/5D_prec/Sonoma/{event}/"
    flood_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma/{event}"
    
    start_time = datetime.strptime("20190226", "%Y%m%d")
    end_time = datetime.strptime("20190301", "%Y%m%d")
    all_days = pd.date_range(start=start_time, end=end_time, freq='d')
    
    dataset = FloodDataset(event, dem_dir, lc_dir, D1_prec_dir, D5_prec_dir, flood_dir, tile_csv, all_days,
                        DEM_mean, DEM_std, day1_prec_mean, day1_prec_std, day5_prec_mean, day5_prec_std,
                        flood_mean=None, flood_std=None, output_geotiff_dir=output_geotiff_dir, return_metadata=True, mode='test')
    # print(data) 
    print(f"Dataset size: {len(dataset)}")  # Should be greater than 0
   
    
    
    
    
        # Example Usage
    # input_image_paths = ["path/to/input_image_1.tif", "path/to/input_image_2.tif"]  # Replace with your file paths
    # output_geotiff_paths = ["output_result_1.tif", "output_result_2.tif"]  # Replace with desired output paths

    # Load dataset
    # dataset = FloodDataset(input_geotiff_paths)

    # Load the trained model
    model_path = f"/p/vast1/lazin1/UNet_trains/Sonoma_multiple_DEM_LC_prec_multiple_Attention/unet_model_fold_7.pt" #f"/p/vast1/lazin1/UNet_trains/Mississippi_20190617_5E5F_non_flood_event_wise/unet_model_fold_5.pt" #"/p/vast1/lazin1/UNet_trains/All_events_shuffled/unet_model_fold_8_9.pt"  # Path to the trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)
    

        
        

    # Test the model and save results
    test_unet_on_dataset(model, dataset, device)
# elif sys.argv[1] == "test":
    
#     model_dir = f"/p/vast1/lazin1/UNet_trains/Mississippi_20190617_C310_non_flood_event_wise"
#     model_path = os.path.join(model_dir, f"unet_model_fold_{5}.pt")  # Change to the fold you want to test
#     model = load_model(model_path)
#     model = model.to(device)
#     for e, EVENT_STR in enumerate(EVENT_STRS):
#         print(EVENT_STR)
#         event_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_{EVENT_STR}.csv" 
#         events = pd.read_csv(event_file, header=None).to_numpy()

#         path_input_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_input_dict.pkl"
#         path_target_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_target_dict.pkl"
#         # output_npy_dir = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/results/npy'
#     # os.makedirs(output_npy_dir, exist_ok=True)
#     # output_str = 'Harvey_D734'
#         test_image_output_array_one_model(model, path_input_dict, path_target_dict, event_file,batch_size=64, device='cuda',EVENT_STR=EVENT_STR, device_ids=device_ids)

# elif sys.argv[1] == "test_event_wise":
#     for e, EVENT_STR in enumerate(EVENT_STRS):
#         event_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_{EVENT_STR}.csv" 
#         events = pd.read_csv(event_file, header=None).to_numpy()
#         model_dir = f"/p/vast1/lazin1/UNet_trains/{EVENT_STR}_event_wise"
#         model_path = os.path.join(model_dir, f"unet_model_fold_{len(events)}.pt")
#         print(model_path)
#         model = load_model(model_path)
#         model = model.to(device)
        
        
#         path_input_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_input_dict.pkl"
#         path_target_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_target_dict.pkl"
#         test_image_output_array_event_wise(model, path_input_dict, path_target_dict, event_file,batch_size=64, device='cuda',EVENT_STR=EVENT_STR, device_ids=device_ids)
        
        
# elif sys.argv[1] == "prec_one_fourth_not_30": #prec_doubled_not_30":
#     for e, EVENT_STR in enumerate(EVENT_STRS):
#         event_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_{EVENT_STR}.csv" 
#         events = pd.read_csv(event_file, header=None).to_numpy()
#         model_dir = f"/p/vast1/lazin1/UNet_trains/{EVENT_STR}_event_wise"
#         model_path = os.path.join(model_dir, f"unet_model_fold_{len(events)}.pt")
#         print(model_path)
#         model = load_model(model_path)
#         model = model.to(device)
        
        
#         path_input_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_input_dict_prec_one_fourth_not_30.pkl" #input_dict_prec_halved_not_30.pkl" #_input_dict_prec_doubled_not_30.pkl"
#         path_target_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_target_dict.pkl"
#         test_image_output_array_event_wise_prec_one_fourth_not_30(model, path_input_dict, path_target_dict, event_file,batch_size=64, device='cuda',EVENT_STR=EVENT_STR, device_ids=device_ids)
    


# Example test data
# Assuming `test_input_data` is a tensor of shape (N, 6, 512, 512) and `test_target_data` of shape (N, 1, 512, 512)
# test_input_data = torch.randn(500, 6, 512, 512)  # Replace with actual test data
# test_target_data = torch.randint(0, 2, (500, 1, 512, 512), dtype=torch.float32)  # Replace with actual test labels

# test_dataset = TensorDataset(test_input_data, test_target_data)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


                


# Run the test
# test_model(model, test_loader)
