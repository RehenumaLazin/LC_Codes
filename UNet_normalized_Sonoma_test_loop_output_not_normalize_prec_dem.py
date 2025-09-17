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

device_ids = [1, 2,3] # List of GPU IDs to use
# import rasterio
# from rasterio.transform import from_origin
# from rasterio import warp


# Define Lazy Loading Dataset
class FloodDataset(Dataset):
    def __init__(self, input_geotiff_paths, label_geotiff_path=None,return_metadata=False):
        """
        Args:
            input_geotiff_paths (list of str): List of 5 paths to input GeoTIFF files.
            label_geotiff_path (str): Path to the label GeoTIFF file.
        """
        self.input_geotiff_paths = input_geotiff_paths
        self.label_geotiff_path = label_geotiff_path
        self.return_metadata = return_metadata

    def __len__(self):
        # Assuming all input GeoTIFFs have the same number of files
        return len(self.input_geotiff_paths[0])

    def __getitem__(self, idx):
        # Read 6 GeoTIFF inputs
        inputs = []
        for path in self.input_geotiff_paths:
            with rasterio.open(path[idx]) as src:
                inputs.append(src.read(1))  # Read the first band
                transform = src.transform
                crs = src.crs

        # Stack the inputs along the channel dimension
        inputs = np.stack(inputs, axis=0)  # Shape: (5, H, W)
        
        # Normalize inputs and labels
        # inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        input_array_normalized = np.empty(inputs.shape, dtype=np.float32)
        input_array_normalized[0,:,:] = (inputs[0,:,:] - np.mean(DEM_mean)) / np.mean(DEM_std)
        # input_array_normalized[1,:,:] =  np.where(inputs[1,:,:] == 11, 1, 0)     
        input_array_normalized[1,:,:] = (inputs[1,:,:] - np.mean(day1_prec_mean)) / np.mean(day1_prec_std)
        input_array_normalized[2,:,:] = (inputs[2,:,:] - np.mean(day5_prec_mean)) / np.mean(day5_prec_std)
        # input_array_normalized[4,:,:] = (inputs[4,:,:] - np.mean(day30_prec_mean)) / np.mean(day30_prec_std)
        # input_array_normalized[4,:,:] = (inputs[4,:,:] - np.mean(flow_mean)) / np.mean(flow_std)
        # array = inputs[5,:,:]
        # array[array == -9999] = 1  # Permanent waterbody is -9999
        # input_array_normalized[5,:,:] = (array - np.mean(SM_mean)) / np.mean(SM_std)
        
        inputs = torch.tensor(input_array_normalized, dtype=torch.float32)
        # print(inputs.shape, 'inputs', self.label_geotiff_path, self.return_metadata)
        if self.label_geotiff_path:
            
            # Read label GeoTIFF
            with rasterio.open(self.label_geotiff_path[idx]) as src:
                label = src.read(1)  # Shape: (H, W)

            # label = (label > 0).astype(np.float32)
            # label = (label - np.mean(FD_mean)) / np.mean(FD_std)
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
            if self.return_metadata:
                return inputs, label, transform, crs
            return inputs, label
            
        if self.return_metadata:
            # print(inputs, transform, crs)
            return inputs, transform, crs

        return inputs           
        # print(inputs, transform, crs)
        # return inputs, transform, crs


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
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        
        features = init_features

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, features)
        self.encoder2 = conv_block(features, features * 2)
        self.encoder3 = conv_block(features * 2, features * 4)
        self.encoder4 = conv_block(features * 4, features * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle = conv_block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = conv_block(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = conv_block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = conv_block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = conv_block(features * 2, features)

        self.out_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        mid = self.middle(self.pool(enc4))

        dec4 = self.upconv4(mid)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

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
    model = UNet(in_channels=3, out_channels=1, init_features=32)
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
    elif len(batch[0]) == 3:  # Testing mode
        inputs = torch.stack([item[0] for item in batch])  # Stack inputs
        transforms = [item[1] for item in batch]  # Extract transforms
        crs_list = [item[2] for item in batch]  # Extract CRS
        return inputs, transforms, crs_list

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

def load_unet_model(model_class, model_path, in_channels=3, out_channels=1, initial_filters=32, device=None):
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
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=int, start_epoch=int, fold=int, checkpoint_path=str, device='cuda', scaler=None):
    model.to(device)
    if scaler is None:
        scaler = GradScaler()
    # train_losses, val_losses, miou = [], [], []
    train_losses, val_losses, mse = [], [], []
    # start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    # if start_epoch > num_epochs:
    #     start_epoch = 0
    # print(start_epoch)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
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
        save_checkpoint(model, optimizer, epoch,fold, checkpoint_path)
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
def k_fold_cross_validation_based_on_events(events,events_file, dataset, num_epochs=int, batch_size=128, model_dir = str, checkpoint_path=str, device='cuda', device_ids=device_ids):
    # input_dict = load_dict(path_input_dict)
    # target_dict = load_dict(path_target_dict)
    # events = pd.read_csv(event_file, header=None).to_numpy()
    
    model = UNet(in_channels=3, out_channels=1, init_features=32).to(device)
    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    
    tile_names = pd.read_csv(events_file)['ras_name'].tolist() 
    # criterion = nn.BCEWithLogitsLoss() #nn.MSELoss())
    criterion = nn.MSELoss()
    
    
    start_epoch, start_fold = load_checkpoint(checkpoint_path, model, optimizer)
    # if start_epoch > num_epochs:
    #     start_epoch = 0
    #     start_fold += 1
    # print(start_epoch, start_fold)
    # all_indices = {'train_indices': [], 'val_indices': []}
    fold_losses = []
    for fold,  _ in enumerate(events, start=start_fold):
        
        event = events[fold]
        print(fold, event)
        
        model_save_path = os.path.join(model_dir, f"unet_model_fold_{fold}.pt") #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/unet_model_fold_{fold + 1}.pt"

        if fold > 0 and os.path.exists(model_save_path):
            
            model = load_model(model_save_path)
            model = model.to(device)
            print(f"model {model_save_path} is loaded")
            print(f"model {event} is already trained, moving to next event")
            
            # continue
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scaler = GradScaler()
            
        # elif fold==0:

        val_keys= [key for key in tile_names if event in key]
        train_keys = [key for key in tile_names if event not in key]
        
        
        train_idx = [
            idx for idx, tile_name in enumerate(tile_names)
            if any(keyword in tile_name for keyword in train_keys)
        ]
        
        val_idx = [
            idx for idx, tile_name in enumerate(tile_names)
            if any(keyword in tile_name for keyword in val_keys)
        ]
        print(len(train_idx), len(val_idx))
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate_fn)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=custom_collate_fn)
        
        if len(train_loader) == 0:
            print("Warning: train_loader is empty. Skipping this fold.")
            continue

        
        
        print(len(train_keys), len(val_keys))






        # optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # start_epoch, start_fold = load_checkpoint(checkpoint_path, model, optimizer)
        # if start_epoch > num_epochs:
        #     start_epoch = 0
        # print(start_epoch)

        # trained_model, train_losses, val_losses, miou = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs,start_epoch, fold, checkpoint_path, device)
        trained_model, train_losses, val_losses, mse = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs,start_epoch, fold, checkpoint_path, device, scaler=scaler)
        
        # Save the model after each fold
        fold_losses.append((train_losses, val_losses))
        model_save_path = os.path.join(model_dir, f"unet_model_fold_{fold+1}.pt")
        torch.save(trained_model.state_dict(), model_save_path) #torch.save(model.state_dict(), str(model_path))
        # np.save(f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/train_val_losses_fold_{fold + 1}.npy" , fold_losses)
        np.save(os.path.join(model_dir, f"train_val_losses_fold_{fold + 1}.npy") , fold_losses)
        # np.save(f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/val_miou_{fold + 1}.npy" , miou)
        np.save(os.path.join(model_dir, f"val_mse_{fold + 1}.npy"),mse)
        print(f"Model for fold {fold + 1} saved as {model_save_path}.\n")
        start_epoch = 0
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

def test_unet_on_dataset(model, dataset, output_geotiff_paths, device):
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
    for i, data in enumerate(dataloader):
        # print(data, 'data')
        inputs, transform, crs = data[0].to(device), data[1][0], data[2][0]
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
            output_geotiff_paths[i],
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

        print(f"Output saved as GeoTIFF: {output_geotiff_paths[i]}")





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




if sys.argv[1] == "train":
# k_fold_groups(dataset, k=5)
# for e, EVENT_STR in enumerate(EVENT_STRS):
    # event_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/EVENT_{EVENT_STR}.csv" 
    event_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma/event2"
    events = [ event for  event in (os.listdir(event_dir)) if os.path.isdir(os.path.join(event_dir, event))]

    model_dir = f"//p/vast1/lazin1/UNet_trains/Sonoma_event2_output_not_normalized_prec_dem_retrain"
    os.makedirs(model_dir,exist_ok=True)
    
    
    checkpoint_path = os.path.join(model_dir,"model_checkpoint.pth")

                
    events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma_event2.csv" #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined_{EVENT_STR}_No_Threshold.csv" #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined.csv"  #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'

    tiles_df = pd.read_csv(events_file)
    tile_names = tiles_df['ras_name'].tolist()  # Assuming the column is named 'tile_name'


    DEM = []
    LC = []
    day1_prec = []
    day5_prec = []
    streamflow = []
    SM = []
    label_geotiff_path = []
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
        
        
    # GeoTIFF paths

        input_geotiff_paths =[
            DEM,
            # LC,
            day1_prec,
            day5_prec,
            # streamflow,
            # SM
                            ]



    # Dataset
    dataset = FloodDataset(input_geotiff_paths, label_geotiff_path)

    k_fold_cross_validation_based_on_events(events, events_file, dataset, num_epochs=50, batch_size=128, model_dir=model_dir, checkpoint_path=checkpoint_path, device=device, device_ids=device_ids)
            




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
    output_geotiff_paths =[]
    output_geotiff_dir =f"/p/vast1/lazin1/UNet_Geotiff_output/Sonoma_event2_prec_dem"
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
        output_geotiff_paths.append(f"{output_geotiff_dir}/{tile_name}")
        
        
    # GeoTIFF paths

    input_geotiff_paths =[
        DEM,
        # LC,
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
    dataset = FloodDataset(input_geotiff_paths, return_metadata=True)
    data = dataset[0]
    print(data) 
    print(f"Dataset size: {len(dataset)}")  # Should be greater than 0
   
    
    
    
    
        # Example Usage
    # input_image_paths = ["path/to/input_image_1.tif", "path/to/input_image_2.tif"]  # Replace with your file paths
    # output_geotiff_paths = ["output_result_1.tif", "output_result_2.tif"]  # Replace with desired output paths

    # Load dataset
    # dataset = FloodDataset(input_geotiff_paths)

    # Load the trained model
    model_path = f"/p/vast1/lazin1/UNet_trains/Sonoma_event2_output_not_normalized_prec_dem_retrain/unet_model_fold_7.pt" #f"/p/vast1/lazin1/UNet_trains/Mississippi_20190617_5E5F_non_flood_event_wise/unet_model_fold_5.pt" #"/p/vast1/lazin1/UNet_trains/All_events_shuffled/unet_model_fold_8_9.pt"  # Path to the trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)
    

        
        

    # Test the model and save results
    test_unet_on_dataset(model, dataset, output_geotiff_paths, device)
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
