import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
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

device_ids = [1, 2,3] # List of GPU IDs to use
# import rasterio
# from rasterio.transform import from_origin
# from rasterio import warp





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
    def __init__(self, in_channels=5, out_channels=1, init_features=32):
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

# Define the test function
def test_model(model, test_loader, device='cuda', threshold=0.5):
    model.to(device)
    model.eval()
    iou_scores = []
    batch_outputs = []
    
    




    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Apply sigmoid if using BCEWithLogitsLoss (for raw logits)
            outputs = torch.sigmoid(outputs)
            # Set a threshold to convert probabilities to binary
            threshold = 0.5
            binary_predictions = (outputs > threshold).float()  # Now we have 0s and 1s
            iou_score = mean_iou(outputs, targets, threshold)
            iou_scores.append(iou_score)
            batch_outputs.append(binary_predictions.cpu())
    batch_outputs = torch.cat(batch_outputs, dim=0)
    average_iou = sum(iou_scores) / len(iou_scores)
    print(f"Mean IoU on test data: {average_iou:.4f}")
    return batch_outputs,average_iou

# Load the saved model for testing
def load_model(model_path, device='cuda'):
    model = UNet(in_channels=5, out_channels=1, init_features=32)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    return model

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

def load_unet_model(model_class, model_path, in_channels=6, out_channels=1, initial_filters=32, device=None):
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
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25, device='cuda'):
    model.to(device)
    scaler = GradScaler()
    train_losses, val_losses, miou = [], [], []
    
    for epoch in range(num_epochs):
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
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        iou_score = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                val_loss += loss.item() * inputs.size(0)
                iou_score += mean_iou(torch.sigmoid(outputs), targets)

        val_loss /= len(val_loader.dataset)
        iou_score /= len(val_loader)
        val_losses.append(val_loss)
        miou.append(iou_score)
        print(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {iou_score:.4f}")
        
    

    return model, train_losses, val_losses, miou
def k_fold_groups(dataset, k=5):
    kfold = KFold(n_splits=k, shuffle=True)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        all_indices = {'train_indices': [], 'val_indices': []}
        all_indices['train_indices'].append(train_idx.tolist())
        all_indices['val_indices'].append(val_idx.tolist())
                
        # Save indices to JSON
        with open(os.path.join(model_dir, str(fold+1) +"_train_val_indices.json"), "w") as f:
            json.dump(all_indices, f)
            
# def k_fold_groups_based_on_events(path_input_dict, path_target_dict, event_file):
#     input_dict = load_dict(path_input_dict)
#     target_dict = load_dict(path_target_dict)
#     events = pd.read_csv(event_file, header=None).to_numpy()
#     # kfold = KFold(n_splits=k, shuffle=True)
    
#     for fold, event in enumerate(events):
#         # all_indices = {'train_indices': [], 'val_indices': []}
#         val_input_array = np.array([data_dict[key] for key in data_dict if key.startswith(events[fold][0].split("/")[-1][:-4])])
#         train_subset = np.array([data_dict[key] for key in data_dict if not key.startswith(events[fold][0].split("/")[-1][:-4])])
        
#         # all_indices['train_indices'].append(train_idx.tolist())
#         # all_indices['val_indices'].append(val_idx.tolist())
                
#         # Save indices to JSON
#         with open(os.path.join(model_dir, str(fold+1) +"_train_val_indices.json"), "w") as f:
#             json.dump(all_indices, f)
            
# K-Fold Cross-Validation
def k_fold_cross_validation_based_on_events(model, path_input_dict, path_target_dict, event_file, num_epochs=100, batch_size=64, device='cuda',device_ids=device_ids):
    input_dict = load_dict(path_input_dict)
    target_dict = load_dict(path_target_dict)
    events = pd.read_csv(event_file, header=None).to_numpy()
    criterion = nn.BCEWithLogitsLoss()
    # all_indices = {'train_indices': [], 'val_indices': []}
    fold_losses = []
    for fold,  event in enumerate(events):

        # all_indices = {'train_indices': [], 'val_indices': []}
        # model_save_path = os.path.join(model_dir, f"unet_model_fold_{fold}.pt") #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/unet_model_fold_{fold + 1}.pt"
        # if os.path.exists(model_save_path):
        #     model = load_model(model_save_path)
        #     model = model.to(device)
        #     print(f"model loaded from {model_save_path}")
            
        # else:
        #     model = UNet(in_channels=5, out_channels=1, init_features=32)
            
        #     # Wrap the model with nn.DataParallel for multi-GPU training
        #     model = nn.DataParallel(model).to(device)
                    
        #     print(f"Fold {fold + 1}/{len(events)}")
                    
        val_keys= [key for key in input_dict if key.startswith(events[fold][0].split("/")[-1][:-4])]
        train_keys = [key for key in input_dict if not key.startswith(events[fold][0].split("/")[-1][:-4])]

        val_input = np.empty((len(val_keys),5,512,512),dtype=np.float32)
        val_target = np.empty((len(val_keys),1,512,512),dtype=np.float32)
        train_input = np.empty((len(train_keys),5,512,512),dtype=np.float32)
        train_target = np.empty((len(train_keys),1,512,512),dtype=np.float32)
        print(len(train_keys), len(val_keys))


        for v, val_key in enumerate(val_keys):
            val_input[v] = input_dict[val_key]
            val_target[v] = target_dict[val_key]
        for t, train_key in enumerate(train_keys):
            train_input[t] = input_dict[train_key]
            train_target[t] = target_dict[train_key]
        
        train_input_array = torch.from_numpy(np.array(train_input))
        train_target_array = torch.from_numpy(np.array(train_target))
        print(train_input.shape[0], train_target.shape[0])
        
        
        val_input_array = torch.from_numpy(np.array(val_input))
        val_target_array = torch.from_numpy(np.array(val_target))
        print( val_input.shape[0], val_target.shape[0])
        
        # input_data = torch.from_numpy(input)
        # target_data = torch.from_numpy(target)

        train_dataset = TensorDataset(train_input_array, train_target_array)
        val_dataset = TensorDataset(val_input_array, val_target_array)
        
        # if os.path.exists(os.path.join(model_dir, str(fold+1) +"_train_val_indices.json")):
        #     with open(os.path.join(model_dir, str(fold+1) +"_train_val_indices.json"), "r") as f:
        #         all_indices = json.load(f)
        #         train_idx = np.array(all_indices['train_indices'][0])
        #         val_idx = np.array(all_indices['val_indices'][0])
        #         print('train_val indices loaded')
        # else:
            
        #     # Save indices for train and validation sets
        #     all_indices['train_indices'].append(train_idx.tolist())
        #     all_indices['val_indices'].append(val_idx.tolist())
            
        #     # Save indices to JSON
        #     with open(os.path.join(model_dir, str(fold+1) +"_train_val_indices.json"), "w") as f:
        #         json.dump(all_indices, f)
        
        # train_subset = Subset(dataset, train_idx)
        # val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        trained_model, train_losses, val_losses, miou = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device)
        
        # Save the model after each fold
        fold_losses.append((train_losses, val_losses))
        model_save_path = os.path.join(model_dir, f"unet_model_fold_{fold+1}.pt")
        torch.save(trained_model.state_dict(), model_save_path) #torch.save(model.state_dict(), str(model_path))
        # np.save(f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/train_val_losses_fold_{fold + 1}.npy" , fold_losses)
        np.save(os.path.join(model_dir, f"train_val_losses_fold_{fold + 1}.npy") , fold_losses)
        # np.save(f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/val_miou_{fold + 1}.npy" , miou)
        np.save(os.path.join(model_dir, f"val_miou_{fold + 1}.npy"),miou)
        print(f"Model for fold {fold + 1} saved as {model_save_path}.\n")
                # Plot losses for this fold
        plt.figure()
        plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Fold {fold + 1} Loss")
        plt.legend()
        plt.savefig(os.path.join(model_dir, f"train_val_losses_fold_{fold + 1}_loss_curve.png"))
        plt.close()



# K-Fold Cross-Validation
def k_fold_cross_validation(dataset, k=5, num_epochs=100, batch_size=64, device='cuda',device_ids=device_ids):
    kfold = KFold(n_splits=k, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    # all_indices = {'train_indices': [], 'val_indices': []}
    fold_losses = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        all_indices = {'train_indices': [], 'val_indices': []}
        model_save_path = os.path.join(model_dir, f"unet_model_fold_{fold + 1}.pt") #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/unet_model_fold_{fold + 1}.pt"
        if os.path.exists(model_save_path):
            continue
        else:
                    
            print(f"Fold {fold + 1}/{k}")
            if os.path.exists(os.path.join(model_dir, str(fold+1) +"_train_val_indices.json")):
                with open(os.path.join(model_dir, str(fold+1) +"_train_val_indices.json"), "r") as f:
                    all_indices = json.load(f)
                    train_idx = np.array(all_indices['train_indices'][0])
                    val_idx = np.array(all_indices['val_indices'][0])
                    print('train_val indices loaded')
            else:
                
                # Save indices for train and validation sets
                all_indices['train_indices'].append(train_idx.tolist())
                all_indices['val_indices'].append(val_idx.tolist())
                
                # Save indices to JSON
                with open(os.path.join(model_dir, str(fold+1) +"_train_val_indices.json"), "w") as f:
                    json.dump(all_indices, f)
            
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            model = UNet(in_channels=5, out_channels=1, init_features=32)
            
            # Wrap the model with nn.DataParallel for multi-GPU training
            model = nn.DataParallel(model).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            trained_model, train_losses, val_losses, miou = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device)
            
            # Save the model after each fold
            fold_losses.append((train_losses, val_losses))
            torch.save(trained_model.state_dict(), model_save_path) #torch.save(model.state_dict(), str(model_path))
            # np.save(f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/train_val_losses_fold_{fold + 1}.npy" , fold_losses)
            np.save(os.path.join(model_dir, f"train_val_losses_fold_{fold + 1}.npy") , fold_losses)
            # np.save(f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/val_miou_{fold + 1}.npy" , miou)
            np.save(os.path.join(model_dir, f"val_miou_{fold + 1}.npy"),miou)
            print(f"Model for fold {fold + 1} saved as {model_save_path}.\n")
                    # Plot losses for this fold
            plt.figure()
            plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
            plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Fold {fold + 1} Loss")
            plt.legend()
            plt.savefig(os.path.join(model_dir, f"train_val_losses_fold_{fold + 1}_loss_curve.png"))
            plt.close()



                
def k_fold_test_output_array(dataset, k=5,  batch_size=64, device='cuda',device_ids=device_ids, output_str=str):


    for fold in range(k):
        print(f"Processing fold {fold + 1}")
        output_arrays = []
        ious = []

        # Load and test each model fold (e.g., model for fold 1)
        model_path = os.path.join(model_dir, f"unet_model_fold_{fold + 1}.pt")  # Change to the fold you want to test
        model = load_model(model_path)
        model = model.to(device)
        print('fold',{fold + 1})
        
        with open(os.path.join(model_dir, str(fold+1) +"_train_val_indices.json"), "r") as f: #os.path.join(output_npy_dir,f"val_result_fold_{fold + 1}.npy" )
            all_indices = json.load(f)
            train_idx = np.array(all_indices['train_indices'][0])
            val_idx = np.array(all_indices['val_indices'][0])
            print(f"Train/validation indices loaded. Validation set size: {len(val_idx)}")
        # Create validation subset and DataLoader
        val_subset = Subset(dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Test the model on the validation set
        output_array, average_iou = test_model(model, test_loader=val_loader, device=device)
        print(f"Output array shape for fold {fold + 1}: {output_array.shape}")
        print(f"Average IoU for fold {fold + 1}: {average_iou}")

        # output_arrays.append(output_array)
        ious.append(average_iou)
        output_npy = os.path.join(output_npy_dir,f"val_result_fold_{fold + 1}_{output_str}.npy" )
        np.save(output_npy, output_array)
        
def test_image_output_array_one_model(model, path_input_dict, path_target_dict, event_file,   batch_size=64, device='cuda',EVENT_STR=str, device_ids=device_ids):
    input_dict = load_dict(path_input_dict)
    target_dict = load_dict(path_target_dict)
    events = pd.read_csv(event_file, header=None).to_numpy()
    
    for event in events:
        event_str= event[0].split("/")[-1][:-4]
        test_keys= [key for key in input_dict if key.startswith(event_str)]
        # train_keys = [key for key in input_dict if not key.startswith(events[fold][0].split("/")[-1][:-4])]
        
        
        # with open('List_Mississippi_20190617_5E5F.csv', 'w', newline='') as csvfile:
        # writer = csv.writer(csvfile)
        # for item in file_lists:
        #     writer.writerow([item])
        os.makedirs(f"{output_npy_dir}/{EVENT_STR}", exist_ok=True)
        # with open(f"{output_npy_dir}/{EVENT_STR}/{event_str}.csv", 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     for row in test_keys:
        #         writer.writerow([row])
        test_input = np.empty((len(test_keys),5,512,512),dtype=np.float32)
        test_target = np.empty((len(test_keys),1,512,512),dtype=np.float32)
        # train_input = np.empty((len(train_keys),5,512,512),dtype=np.float32)
        # train_target = np.empty((len(train_keys),1,512,512),dtype=np.float32)
        # print(len(train_keys), len(val_keys))

        with open(f"{output_npy_dir}/{EVENT_STR}/{event_str}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            for t, test_key in enumerate(test_keys):
                test_input[t] = input_dict[test_key]
                test_target[t] = target_dict[test_key]
                writer.writerow([test_key])
        # for t, train_key in enumerate(train_keys):
        #     train_input[t] = input_dict[train_key]
        #     train_target[t] = target_dict[train_key]
        
        # train_input_array = torch.from_numpy(np.array(test_input))
        # train_target_array = torch.from_numpy(np.array(test_target))
        
        
        test_input_array = torch.from_numpy(np.array(test_input))
        test_target_array = torch.from_numpy(np.array(test_target))
        print(test_input.shape[0], test_target.shape[0])
        
        # input_data = torch.from_numpy(input)
        # target_data = torch.from_numpy(target)

        # train_dataset = TensorDataset(train_input_array, train_target_array)
        test_dataset = TensorDataset(test_input_array, test_target_array)
        
        
        
        # model_path = os.path.join(model_dir, f"unet_model_fold_{fold}.pt")  # Change to the fold you want to test
        # model = load_model(model_path)
        # model = model.to(device)
    #     print('fold',{fold + 1})
        
    #     with open(os.path.join('/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet', str(fold+1) +"_train_val_indices.json"), "r") as f:
    #         all_indices = json.load(f)
    #         train_idx = np.array(all_indices['train_indices'][0])
    #         val_idx = np.array(all_indices['val_indices'][0])
    #         print(f"Train/validation indices loaded. Validation set size: {len(val_idx)}")
    #     # Create validation subset and DataLoader
    #     val_subset = Subset(dataset, val_idx)
        data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Test the model on the validation set
        output_array, average_iou = test_model(model, test_loader=data_loader, device=device)
        print(f"Output array shape for fold {fold }: {output_array.shape}")
        print(f"Average IoU for fold {fold }: {average_iou}")

        # output_arrays.append(output_array)
        # ious.append(average_iou)
        output_npy = f"{output_npy_dir}/{EVENT_STR}/{event_str}.npz"
        
        np.savez(output_npy, output_array, average_iou)
        print(f"{event_str}.npz saved")



def test_image_output_array_event_wise(model, path_input_dict, path_target_dict, event_file, fold=int,  batch_size=64, device='cuda',EVENT_STR=str, device_ids=device_ids):
    input_dict = load_dict(path_input_dict)
    target_dict = load_dict(path_target_dict)
    events = pd.read_csv(event_file, header=None).to_numpy()
    
    for event in events:
        event_str= event[0].split("/")[-1][:-4]
        test_keys= [key for key in input_dict if key.startswith(event_str)]

        os.makedirs(f"{output_npy_dir}/{EVENT_STR}", exist_ok=True)

        test_input = np.empty((len(test_keys),5,512,512),dtype=np.float32)
        test_target = np.empty((len(test_keys),1,512,512),dtype=np.float32)


        with open(f"{output_npy_dir}/{EVENT_STR}/{event_str}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            for t, test_key in enumerate(test_keys):
                test_input[t] = input_dict[test_key]
                test_target[t] = target_dict[test_key]
                writer.writerow([test_key])

        
        
        test_input_array = torch.from_numpy(np.array(test_input))
        test_target_array = torch.from_numpy(np.array(test_target))
        print(test_input.shape[0], test_target.shape[0])
        
        test_dataset = TensorDataset(test_input_array, test_target_array)
        
        
        
        # model_path = os.path.join(model_dir, f"unet_model_fold_{fold}.pt")  # Change to the fold you want to test
        # model = load_model(model_path)
        # model = model.to(device)
    #     print('fold',{fold + 1})
        
    #     with open(os.path.join('/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet', str(fold+1) +"_train_val_indices.json"), "r") as f:
    #         all_indices = json.load(f)
    #         train_idx = np.array(all_indices['train_indices'][0])
    #         val_idx = np.array(all_indices['val_indices'][0])
    #         print(f"Train/validation indices loaded. Validation set size: {len(val_idx)}")
    #     # Create validation subset and DataLoader
    #     val_subset = Subset(dataset, val_idx)
        data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Test the model on the validation set
        output_array, average_iou = test_model(model, test_loader=data_loader, device=device)
        print(f"Output array shape for fold {fold }: {output_array.shape}")
        print(f"Average IoU for fold {fold }: {average_iou}")

        # output_arrays.append(output_array)
        # ious.append(average_iou)
        output_npy = f"{output_npy_dir}/{EVENT_STR}/{event_str}_event_wise.npz"
        
        np.savez(output_npy, output_array, average_iou)
        print(f"{event_str}.npz saved")



def test_image_output_array_event_wise_prec_one_fourth_not_30(model, path_input_dict, path_target_dict, event_file, fold=int,  batch_size=64, device='cuda',EVENT_STR=str, device_ids=device_ids):
    input_dict = load_dict(path_input_dict)
    target_dict = load_dict(path_target_dict)
    events = pd.read_csv(event_file, header=None).to_numpy()
    
    for event in events:
        event_str= event[0].split("/")[-1][:-4]
        test_keys= [key for key in input_dict if key.startswith(event_str)]

        os.makedirs(f"{output_npy_dir}/{EVENT_STR}", exist_ok=True)

        test_input = np.empty((len(test_keys),5,512,512),dtype=np.float32)
        test_target = np.empty((len(test_keys),1,512,512),dtype=np.float32)


        with open(f"{output_npy_dir}/{EVENT_STR}/{event_str}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            for t, test_key in enumerate(test_keys):
                test_input[t] = input_dict[test_key]
                test_target[t] = target_dict[test_key]
                writer.writerow([test_key])

        
        # test_input[:,2,:,:] = test_input[:,2,:,:]*2
        # test_input[:,3,:,:] = test_input[:,3,:,:]*2
        # test_input[:,4,:,:] = test_input[:,4,:,:]*1.5
        test_input_array = torch.from_numpy(np.array(test_input))
        test_target_array = torch.from_numpy(np.array(test_target))
        print(test_input.shape[0], test_target.shape[0])
        
        test_dataset = TensorDataset(test_input_array, test_target_array)
        
        
        
        # model_path = os.path.join(model_dir, f"unet_model_fold_{fold}.pt")  # Change to the fold you want to test
        # model = load_model(model_path)
        # model = model.to(device)
    #     print('fold',{fold + 1})
        
    #     with open(os.path.join('/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet', str(fold+1) +"_train_val_indices.json"), "r") as f:
    #         all_indices = json.load(f)
    #         train_idx = np.array(all_indices['train_indices'][0])
    #         val_idx = np.array(all_indices['val_indices'][0])
    #         print(f"Train/validation indices loaded. Validation set size: {len(val_idx)}")
    #     # Create validation subset and DataLoader
    #     val_subset = Subset(dataset, val_idx)
        data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Test the model on the validation set
        output_array, average_iou = test_model(model, test_loader=data_loader, device=device)
        print(f"Output array shape for fold {fold }: {output_array.shape}")
        print(f"Average IoU for fold {fold }: {average_iou}")

        # output_arrays.append(output_array)
        # ious.append(average_iou)
        output_npy = f"{output_npy_dir}/{EVENT_STR}/{event_str}_event_wise_prec_one_fourth_not_30.npz" #_input_dict_prec_halved_not_30.pkl
        
        np.savez(output_npy, output_array, average_iou)
        print(f"{event_str}.npz saved")





# def test_image_output_array(dataset, k=5, fold=int,  batch_size=64, device='cuda',device_ids=device_ids, output_str=str):

#     model_path = os.path.join(model_dir, f"unet_model_fold_{fold}.pt")  # Change to the fold you want to test
#     model = load_model(model_path)
#     model = model.to(device)
# #     print('fold',{fold + 1})
    
# #     with open(os.path.join('/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet', str(fold+1) +"_train_val_indices.json"), "r") as f:
# #         all_indices = json.load(f)
# #         train_idx = np.array(all_indices['train_indices'][0])
# #         val_idx = np.array(all_indices['val_indices'][0])
# #         print(f"Train/validation indices loaded. Validation set size: {len(val_idx)}")
# #     # Create validation subset and DataLoader
# #     val_subset = Subset(dataset, val_idx)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     # Test the model on the validation set
#     output_array, average_iou = test_model(model, test_loader=data_loader, device=device)
#     print(f"Output array shape for fold {fold }: {output_array.shape}")
#     print(f"Average IoU for fold {fold }: {average_iou}")

#     # output_arrays.append(output_array)
#     # ious.append(average_iou)
#     output_npy = os.path.join(output_npy_dir,f"test_result_{output_str}_fold_{fold}.npy" )
#     np.save(output_npy, output_array)

#     # return output_arrays, ious
#                 # for idx in val_idx:
#             #     reference_tiff_path = os.path.join(reference_raster_dir,reference_raster_files[idx][1])
#             #     output_npy = os.path.join(output_npy_dir,reference_raster_files[idx][1][:-4] + '.npy')
#             #     np.save
#                 # write_geotiff_from_array(output_GeoTIFF, output_array, reference_tiff_path)





# All Dataset
# input = np.load('/p/vast1/lazin1/UNet_inputs/input_data_normalized_Harvey_D734.npy') #/p/vast1/lazin1/UNet_inputs/input_data_normalized_Mississippi_20190617_5E5F_flood.npy') #'/p/vast1/lazin1/UNet_inputs/input_data_normalized_Mississippi_20190617_3AC6.npy') #'/p/vast1/lazin1/UNet_inputs/input_data_normalized_20190617_5E5F_non_flood_5_var.npy') #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_input_data_combined.npy')
# target = np.load('/p/vast1/lazin1/UNet_inputs/target_data_Harvey_D734.npy') #/p/vast1/lazin1/UNet_inputs/target_data_Mississippi_20190617_5E5F_flood.npy') #'/p/vast1/lazin1/UNet_inputs/target_data_Mississippi_20190617_3AC6.npy')  #/p/vast1/lazin1/UNet_inputs/target_data_20190617_5E5F_non_flood.npy.npy') #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/target_data_combined.npy')
# # input_data = torch.randn(4275, 5, 512, 512)  # Replace with actual input data
# # target_data = torch.randint(0, 2, (4275, 1, 512, 512), dtype=torch.float32)  # Replace with actual target data
# input_data = torch.from_numpy(input)
# target_data = torch.from_numpy(target)

# dataset = TensorDataset(input_data, target_data)
# EVENT_STR = "Mississippi_20190617_3AC6_non_flood" #"Mississippi_20190617_42BF_non_flood      unfinished "  #"Mississippi_20190617_9D85_non_flood" #"Mississippi_20190617_5E5F_non_flood"
# EVENT_STRS = ["Mississippi_20190617_5E5F_non_flood", "Mississippi_20190617_9D85_non_flood", "Mississippi_20190617_3AC6_non_flood", "Harvey_20170829_D734_non_flood","Harvey_20170831_9366_non_flood" , "Harvey_20170831_0776_non_flood", "Harvey_20170829_B8C4_non_flood", "Harvey_20170829_3220_non_flood", "Florence_20180919_B86C_non_flood", "Mississippi_20190617_C310_non_flood"] #"Mississippi_20190617_B9D7_non_flood"  #"Florence_20180919_374E_non_flood"
# EVENT_STRS = ["Florence_20180919_374E_non_flood", "Florence_20180919_B86C_non_flood"]

EVENT_STRS = ["Mississippi_20190617_3AC6_non_flood"]


events_file = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'
combined_df = pd.read_csv(events_file, header=None) 
events = [i.split("/")[-1][:-4] for i in combined_df[0]]
# print(events)

# EVENT_STRS = ["Mississippi_20190617_5E5F_non_flood", "Mississippi_20190617_9D85_non_flood", "Mississippi_20190617_3AC6_non_flood", "Harvey_20170829_D734_non_flood","Harvey_20170831_9366_non_flood" , "Harvey_20170831_0776_non_flood", "Harvey_20170829_B8C4_non_flood", "Harvey_20170829_3220_non_flood", "Florence_20180919_B86C_non_flood", "Mississippi_20190617_C310_non_flood"] #"Mississippi_20190617_B9D7_non_flood"  #"Florence_20180919_374E_non_flood"


# model_dir = f"/p/vast1/lazin1/UNet_trains/{EVENT_STR}_event_wise"
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
output_npy_dir = '/p/vast1/lazin1/UNet_npy_output'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    



if sys.argv[1] == "train":
    # k_fold_groups(dataset, k=5)
    for e, EVENT_STR in enumerate(EVENT_STRS):
        event_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_{EVENT_STR}.csv" 
        events = pd.read_csv(event_file, header=None).to_numpy()
        model_dir = f"/p/vast1/lazin1/UNet_trains/{EVENT_STR}_event_wise"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_save_path = os.path.join(model_dir, f"unet_model_fold_{len(events)}.pt") #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/unet_model_fold_{fold + 1}.pt"
        if os.path.exists(model_save_path):
            print(f"model {EVENT_STR} is already trained, moving to next event")
            continue
            # model = load_model(model_save_path)
            # model = model.to(device)
            # print(f"model loaded from {model_save_path}")
            
        else:
            print(e,EVENT_STR)
            event_str = EVENT_STRS[e-1]
            model_dir = f"/p/vast1/lazin1/UNet_trains/{event_str}_event_wise"
            events = pd.read_csv(f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_{event_str}.csv", header=None).to_numpy()
            model_save_path = os.path.join(model_dir, f"unet_model_fold_{len(events)}.pt")
            model = load_model(model_save_path)
            model = model.to(device)
            print(f"model {model_save_path} is loaded")
            # model = UNet(in_channels=5, out_channels=1, init_features=32)
            
            # Wrap the model with nn.DataParallel for multi-GPU training
            # model = nn.DataParallel(model).to(device)
            model_dir = f"/p/vast1/lazin1/UNet_trains/{EVENT_STR}_event_wise"
                    
            
            
            path_input_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_input_dict.pkl"
            path_target_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_target_dict.pkl"
            #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/flood_img_list_flood_non_flood_test.csv"

            k_fold_cross_validation_based_on_events(model,path_input_dict, path_target_dict, event_file, num_epochs=50, batch_size=128, device=device, device_ids=device_ids)
# elif sys.argv[1] == "create_groups":
#     k_fold_groups(dataset, k=5)
# elif sys.argv[1] == "evaluate":
#     k_fold_test_output_array(dataset, k=5,  batch_size=64, device='cuda',device_ids=device_ids)
elif sys.argv[1] == "test":
    
    model_dir = f"/p/vast1/lazin1/UNet_trains/Mississippi_20190617_C310_non_flood_event_wise"
    model_path = os.path.join(model_dir, f"unet_model_fold_{5}.pt")  # Change to the fold you want to test
    model = load_model(model_path)
    model = model.to(device)
    for e, EVENT_STR in enumerate(EVENT_STRS):
        print(EVENT_STR)
        event_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_{EVENT_STR}.csv" 
        events = pd.read_csv(event_file, header=None).to_numpy()

        path_input_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_input_dict.pkl"
        path_target_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_target_dict.pkl"
        # output_npy_dir = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/results/npy'
    # os.makedirs(output_npy_dir, exist_ok=True)
    # output_str = 'Harvey_D734'
        test_image_output_array_one_model(model, path_input_dict, path_target_dict, event_file,batch_size=64, device='cuda',EVENT_STR=EVENT_STR, device_ids=device_ids)

elif sys.argv[1] == "test_event_wise":
    for e, EVENT_STR in enumerate(EVENT_STRS):
        event_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_{EVENT_STR}.csv" 
        events = pd.read_csv(event_file, header=None).to_numpy()
        model_dir = f"/p/vast1/lazin1/UNet_trains/{EVENT_STR}_event_wise"
        model_path = os.path.join(model_dir, f"unet_model_fold_{len(events)}.pt")
        print(model_path)
        model = load_model(model_path)
        model = model.to(device)
        
        
        path_input_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_input_dict.pkl"
        path_target_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_target_dict.pkl"
        test_image_output_array_event_wise(model, path_input_dict, path_target_dict, event_file,batch_size=64, device='cuda',EVENT_STR=EVENT_STR, device_ids=device_ids)
        
        
elif sys.argv[1] == "prec_one_fourth_not_30": #prec_doubled_not_30":
    for e, EVENT_STR in enumerate(EVENT_STRS):
        event_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_{EVENT_STR}.csv" 
        events = pd.read_csv(event_file, header=None).to_numpy()
        model_dir = f"/p/vast1/lazin1/UNet_trains/{EVENT_STR}_event_wise"
        model_path = os.path.join(model_dir, f"unet_model_fold_{len(events)}.pt")
        print(model_path)
        model = load_model(model_path)
        model = model.to(device)
        
        
        path_input_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_input_dict_prec_one_fourth_not_30.pkl" #input_dict_prec_halved_not_30.pkl" #_input_dict_prec_doubled_not_30.pkl"
        path_target_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_target_dict.pkl"
        test_image_output_array_event_wise_prec_one_fourth_not_30(model, path_input_dict, path_target_dict, event_file,batch_size=64, device='cuda',EVENT_STR=EVENT_STR, device_ids=device_ids)
    


# Example test data
# Assuming `test_input_data` is a tensor of shape (N, 6, 512, 512) and `test_target_data` of shape (N, 1, 512, 512)
# test_input_data = torch.randn(500, 6, 512, 512)  # Replace with actual test data
# test_target_data = torch.randint(0, 2, (500, 1, 512, 512), dtype=torch.float32)  # Replace with actual test labels

# test_dataset = TensorDataset(test_input_data, test_target_data)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


                


# Run the test
# test_model(model, test_loader)
