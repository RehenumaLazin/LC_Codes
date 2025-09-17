import os
import argparse
import glob
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
torch.cuda.empty_cache()

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

    def _generate_samples(self):
        samples = []
        for tile in self.tiles:
            event = tile.split("/")[-1].split("_")[0]
            tile_suffix = tile.split("Sonoma")[-1]
            for i in range(self.seq_len - 1, len(self.hours)):
                seq_hours = self.hours[i - self.seq_len + 1:i + 1]
                target_hour = self.hours[i]
                if all(os.path.exists(os.path.join(self.precip_dir, f"1H_prec_{event}_{h.strftime('%Y%m%d%H')}_Sonoma{tile_suffix}")) for h in seq_hours) and \
                   os.path.exists(os.path.join(self.flood_dir, f"{event}_{target_hour.strftime('%Y%m%d%H')}_Sonoma{tile_suffix}")):
                    samples.append((tile, seq_hours, target_hour))
        return samples

    def _read_raster(self, path):
        with rasterio.open(path) as src:
            array = src.read(1)
            transform = src.transform
            crs = src.crs
        return array, transform, crs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tile, seq_hours, target_hour = self.samples[idx]
        tile_suffix = tile.split("Sonoma")[-1]
        event = tile.split("/")[-1].split("_")[0]

        dem_path = os.path.join(self.dem_dir, f"Sonoma{tile_suffix}")
        lc_path = os.path.join(self.lc_dir, f"LC_Sonoma{tile_suffix}")
        dem, transform, crs = self._read_raster(dem_path)
        lc, _, _ = self._read_raster(lc_path)

        dem_norm = (dem - self.DEM_mean) / self.DEM_std
        lc_binary = np.where(lc == 11, 1, 0)  # Optional: one-hot or binary

        precip_seq = []
        for h in seq_hours:
            p_path = os.path.join(self.precip_dir, f"1H_prec_{event}_{h.strftime('%Y%m%d%H')}_Sonoma{tile_suffix}")
            p, _, _ = self._read_raster(p_path)
            p_norm = (p - self.precip_mean) / self.precip_std
            precip_seq.append(p_norm)
        precip_seq = np.stack(precip_seq, axis=0)

        # Final shape: (channels, H, W)
        inputs = np.concatenate([
            dem_norm[np.newaxis, :, :],
            lc_binary[np.newaxis, :, :],
            precip_seq
        ], axis=0).astype(np.float32)

        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

        # Load label
        label_path = os.path.join(self.flood_dir, f"{event}_{target_hour.strftime('%Y%m%d%H')}_Sonoma{tile_suffix}")
        label, _, _ = self._read_raster(label_path)

        if self.flood_mean is not None and self.flood_std is not None and self.mode == 'train':
            label = (label - self.flood_mean) / self.flood_std

        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        if self.return_metadata:
            return inputs_tensor, label_tensor, transform, crs
        return inputs_tensor, label_tensor


# --- Model ---


# --- Basic ConvBlock ---


# --- Basic ConvBlock ---
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

# --- UNet-LSTM Model ---
class UNetLSTM(nn.Module):
    def __init__(self, in_static_ch=2, in_seq_ch=1, seq_len=10, hidden_dim=512):
        super().__init__()
        self.seq_len = seq_len

        # Encoder layers for sequential input
        self.enc1 = ConvBlock(in_seq_ch, 32)  # -> 512x512
        self.enc2 = ConvBlock(32, 64)         # -> 256x256
        self.enc3 = ConvBlock(64, 128)        # -> 128x128
        self.pool = nn.MaxPool2d(2)

        # Static encoder
        self.static_encoder = ConvBlock(in_static_ch, 64)  # -> 512x512

        # LSTM over bottleneck features (128x128)
        self.lstm = nn.LSTM(input_size=128 * 128 * 128, hidden_size=hidden_dim, batch_first=True)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(hidden_dim + 64, 64, 2, stride=2)  # -> 256x256
        self.dec1 = ConvBlock(64 + 128, 64)  # skip from enc3
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # -> 512x512
        self.dec2 = ConvBlock(32 + 64, 32)  # skip from static
        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, static_input, seq_input):
        B, T, C, H, W = seq_input.shape
        assert H == 512 and W == 512, "Input must be 512x512"

        # Encode static input
        static_feat = self.static_encoder(static_input)  # (B, 64, 512, 512)
        static_ds = F.avg_pool2d(static_feat, 8)          # (B, 64, 64, 64)

        bottlenecks = []
        skip_enc3 = None  # Save for decoder

        for t in range(T):
            xt = seq_input[:, t]              # (B, 1, 512, 512)
            x1 = self.enc1(xt)                # (B, 32, 512, 512)
            x2 = self.pool(x1)                # (B, 32, 256, 256)
            x2 = self.enc2(x2)                # (B, 64, 256, 256)
            x3 = self.pool(x2)                # (B, 64, 128, 128)
            x3 = self.enc3(x3)                # (B, 128, 128, 128)
            x3_pooled = F.adaptive_avg_pool2d(x3, (16, 16)) # Or even (8, 8)
            bottlenecks.append(x3_pooled)
            if t == T - 1:
                skip_enc3 = x3                # For skip connection

        bottlenecks = torch.stack(bottlenecks, dim=1)     # (B, T, 128, 128, 128)
        bottlenecks_flat = bottlenecks.view(B, T, -1)     # (B, T, 128*128*128)

        lstm_out, _ = self.lstm(bottlenecks_flat)
        last_output = lstm_out[:, -1]                     # (B, hidden_dim)
        lstm_feat = last_output.view(B, -1, 1, 1).expand(-1, -1, 64, 64)  # (B, hidden_dim, 64, 64)

        # Concatenate LSTM output and static features
        x = torch.cat([lstm_feat, static_ds], dim=1)      # (B, hidden_dim + 64, 64, 64)

        # Decoder
        x = self.upconv1(x)                               # (B, 64, 128, 128)
        x = torch.cat([x, skip_enc3], dim=1)              # (B, 192, 128, 128)
        x = self.dec1(x)                                  # (B, 64, 128, 128)

        x = self.upconv2(x)                               # (B, 32, 256, 256)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        x = torch.cat([x, static_feat], dim=1)            # (B, 32+64, 512, 512)
        x = self.dec2(x)                                  # (B, 32, 512, 512)

        return self.final_conv(x).squeeze(1)              # (B, 512, 512)



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

        for inputs, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
            static = inputs[:, :2] 
            seq = inputs[:, 2:].unsqueeze(2) 
            static, seq, target = static.to(device), seq.to(device), target.to(device)
            optimizer.zero_grad()
            
            with autocast():
                output = model(static, seq)
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
                running_loss += loss.item() * inputs.size(0)
            else:
                print(f"Skipping step due to invalid loss at epoch {epoch}")
            
            
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")
        
        
        model.eval()
        val_loss = 0.0
        mse_score = 0.0
        
        with torch.no_grad():
            for inputs, target in val_loader:
                static = inputs[:, :2] 
                seq = inputs[:, 2:].unsqueeze(2) 
                static, seq, target = static.to(device), seq.to(device), target.to(device)
                with autocast():
                    output = model(static, seq)
                    
                    loss = criterion(output, target)
                    
                val_loss += loss.item() * inputs.size(0)
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
    # model = nn.DataParallel(model).to(device)
    model = model.to(device)
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
    # model = nn.DataParallel(model).to(device)
    model = model.to(device)
    checkpoint_path = os.path.join(model_dir, "model_checkpoint.pth")
    
    # Extract all unique days (dates only)
    all_dates = sorted(list({s[2].date() for s in dataset.samples}))
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

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, collate_fn=custom_collate_fn)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, collate_fn=custom_collate_fn)

        model = UNetLSTM().to(device)
        # model = nn.DataParallel(model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = GradScaler()
        criterion = nn.MSELoss()
        
        model_save_path = os.path.join(model_dir, f"unet_model_fold_{fold_idx}.pt") #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/unet_model_fold_{fold + 1}.pt"

        if fold_idx > 0 and os.path.exists(model_save_path):
            
            model = load_model(model_save_path)
            # model = nn.DataParallel(model).to(device)
            model = model.to(device)

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
    for data in tqdm(dataloader, desc="Testing"):
        # print(data, 'data')
        # inputs, transform, crs = data[0].to(device), data[1][0], data[2][0] #static_input, precip_tensor, label_tensor, transform, crs
        static_input, precip_tensor, output_geotiff_path, transform, crs = data[0], data[1], data[2][0],data[3][0] , data[4][0] 
        with torch.no_grad():
            outputs = model(static_input, precip_tensor)  # Add batch dimension
            # outputs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            # print(outputs.shape)
            outputs = outputs.squeeze().cpu().numpy()
            # print(outputs.shape)
            
            
            # std = np.mean(FD_std)
            # mean = np.mean(FD_mean)
            # outputs_denorm = outputs * std + mean

        # Binarize the output (optional)
        # output_binary = (outputs > 0.5).astype(np.float32)

        # Save the result as a GeoTIFF
        with rasterio.open(
            output_geotiff_path,
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

        print(f"Output saved as GeoTIFF: {output_geotiff_path}") 
# --- Main ---
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("mode", choices=["train", "test"])
#     parser.add_argument("--start_epoch", type=int, default=0)
#     parser.add_argument("--num_epochs", type=int, default=10)
#     parser.add_argument("--checkpoint", type=str, default=None)
#     parser.add_argument("--k_folds", type=int, default=5)
#     args = parser.parse_args()

 

    # print("K-Fold Cross Validation Completed")
    # for f, d in fold_results.items():
    #     print(f"Fold {f+1} validated on {d}")
        
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("mode", choices=["train", "test"])
#     parser.add_argument("--start_epoch", type=int, default=0)
#     parser.add_argument("--num_epochs", type=int, default=10)
#     parser.add_argument("--checkpoint", type=str, default=None)
#     args = parser.parse_args()

#     dataset = FloodDataset(dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, all_hours, sequence_length,
#                     DEM_mean, DEM_std, precip_mean, precip_std,
#                     flood_mean=None, flood_std=None, return_metadata=False, mode='train')
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=(args.mode == "train"), num_workers=4,  collate_fn=custom_collate_fn)

#     model = UNetLSTM().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     criterion = nn.MSELoss()

#     if args.mode == "train":
#         if args.start_epoch > 0 and args.checkpoint:
#             checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
#             if os.path.exists(checkpoint_path):
#                 ckpt = torch.load(checkpoint_path, map_location=device)
#                 model.load_state_dict(ckpt['model_state_dict'])
#                 optimizer.load_state_dict(ckpt['optimizer_state_dict'])
#                 print(f"Resumed from checkpoint {checkpoint_path}")
#         for epoch in range(args.start_epoch + 1, args.start_epoch + args.num_epochs + 1):
#             train(model, dataloader, optimizer, criterion, epoch, checkpoint_dir)

#     elif args.mode == "test":
#         if not args.checkpoint:
#             raise ValueError("Please provide a checkpoint path with --checkpoint for testing.")
#         checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
#         if not os.path.exists(checkpoint_path):
#             raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")
#         ckpt = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(ckpt['model_state_dict'])
#         print(f"Loaded model from {checkpoint_path}")
#         test(model, dataloader, criterion)


    
    

# event_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/event2"
events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma_hourly_event2.csv"
tile_csv = events_file

model_dir = f"/p/vast1/lazin1/UNet_trains/Sonoma_event2_output_prec_dem_LC_HOURLY_UNET_LSTM_complex"
os.makedirs(model_dir, exist_ok=True)



# tiles_df = pd.read_csv(events_file)
# tile_names = tiles_df['ras_name'].tolist()


dem_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma"
lc_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_LC/Sonoma"
# flood_paths.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/{event}/{tile_name}")

precip_dir = "/p/vast1/lazin1/UNet_inputs/Geotiff_var/1H_prec/Sonoma/event2"
flood_dir = "/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/event2"

# --- Time setup ---
sequence_length = 10
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

    kfold_train(dataset, k_folds=3, batch_size=4, model_dir=model_dir, num_epochs=10)




# python script.py train --start_epoch 0 --num_epochs 10
# python script.py test --checkpoint model_epoch_10.pt
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("mode", choices=["train", "test"])
#     parser.add_argument("--start_epoch", type=int, default=0)
#     parser.add_argument("--num_epochs", type=int, default=10)
#     parser.add_argument("--checkpoint", type=str, default=None)
#     parser.add_argument("--k_folds", type=int, default=5)
#     args = parser.parse_args()

#     # Prepare full dataset once
#     dataset = FloodDataset(dem_dir, lc_dir, precip_dir, flood_dir, tile_csv, all_hours, sequence_length,
#                            DEM_mean, DEM_std, H1_prec_mean, H1_prec_std,
#                            flood_mean=None, flood_std=None, return_metadata=False, mode='train')

#     # Extract all unique days (dates only)
#     all_dates = sorted(list({s[2].date() for s in dataset.samples}))
#     assert len(all_dates) >= args.k_folds, "Not enough unique days for the requested number of folds."

#     fold_results = {}

#     for fold_idx in range(args.k_folds):
#         val_day = all_dates[fold_idx]

#         train_indices = [i for i, s in enumerate(dataset.samples) if s[2].date() != val_day]
#         val_indices = [i for i, s in enumerate(dataset.samples) if s[2].date() == val_day]

#         train_sampler = SubsetRandomSampler(train_indices)
#         val_sampler = SubsetRandomSampler(val_indices)

#         train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler, num_workers=4, collate_fn=custom_collate_fn)
#         val_loader = DataLoader(dataset, batch_size=4, sampler=val_sampler, num_workers=4, collate_fn=custom_collate_fn)

#         model = UNetLSTM().to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#         criterion = nn.MSELoss()

#         print(f"Starting Fold {fold_idx+1}/{args.k_folds}, Validation Day: {val_day}")

#         for epoch in range(args.start_epoch + 1, args.start_epoch + args.num_epochs + 1):
#             train(model, train_loader, optimizer, criterion, epoch, checkpoint_dir)
#             test(model, val_loader, criterion)

#         fold_results[fold_idx] = val_day

#     print("K-Fold Cross Validation Completed")
#     for f, d in fold_results.items():
#         print(f"Fold {f+1} validated on {d}")




##########UNEt-LSTM


# import torch
# import torch.nn as nn
# import torch.nn.functional as F


