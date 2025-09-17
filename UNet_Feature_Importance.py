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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [1, 2,3] # List of GPU IDs to use
output_npy_dir = '/p/vast1/lazin1/UNet_npy_output_feature_importance'

if not os.path.exists(output_npy_dir):
    os.makedirs(output_npy_dir)
# # Load the pre-trained U-Net model
# model = UNet(in_channels=6, out_channels=1)  # Adjust input/output channels as needed
# model.load_state_dict(torch.load("unet_model.pt"))  # Load your trained model
# model.eval()  # Set the model to evaluation mode

# # Define the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define the dataset and data loader
# # Replace input_data and target_data with actual data tensors
# input_data = torch.randn(4275, 6, 512, 512)  # 4275 samples, 6 features, 512x512 images
# target_data = torch.randint(0, 2, (4275, 1, 512, 512), dtype=torch.float32)
# dataset = TensorDataset(input_data, target_data)
# data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
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

def test_model(model, test_loader, device='cuda', threshold=0.5):
    # model.to(device)
    # model.eval()
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

def load_model(model_path, device='cuda'):
    model = UNet(in_channels=5, out_channels=1, init_features=32)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
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

# mIoU metric function
def mean_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).clamp(0, 1).sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def test_image_output_array_event_wise(model, path_input_dict, path_target_dict, event_file, fold=int,  batch_size=64, device='cuda',EVENT_STR=str, device_ids=device_ids):

        # Test the model on the validation set
        output_array, average_iou = test_model(model, test_loader=data_loader, device=device)
        print(f"Output array shape for fold {fold }: {output_array.shape}")
        print(f"Average IoU for fold {fold }: {average_iou}")

        # output_arrays.append(output_array)
        # ious.append(average_iou)
        output_npy = f"{output_npy_dir}/{EVENT_STR}/{event_str}_event_wise.npz"
        
        np.savez(output_npy, output_array, average_iou)
        print(f"{event_str}.npz saved")



# # Test with all features
# def test_model(loader, model, device):
#     model.eval()
#     total_iou = 0.0
#     with torch.no_grad():
#         for inputs, targets in loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             total_iou += mean_iou(outputs, targets)
#     return total_iou / len(loader)

# Baseline performance with all features
# print("Evaluating baseline performance with all features...")
# baseline_iou = test_model(data_loader, model, device)
# print(f"Baseline mIoU: {baseline_iou:.4f}")

EVENT_STRS = ["Harvey_20170829_D734_non_flood"]
for e, EVENT_STR in enumerate(EVENT_STRS):
    if not os.path.exists(f"{output_npy_dir}/{EVENT_STR}"):
        os.makedirs(f"{output_npy_dir}/{EVENT_STR}")
    event_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_{EVENT_STR}.csv" 
    events = pd.read_csv(event_file, header=None).to_numpy()
    model_dir = f"/p/vast1/lazin1/UNet_trains/{EVENT_STR}_event_wise"
    model_path = os.path.join(model_dir, f"unet_model_fold_{len(events)}.pt")
    print(model_path)
    model = load_model(model_path)
    model = model.to(device)
    
    
    path_input_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_input_dict.pkl"
    path_target_dict = f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_target_dict.pkl"
    
    
    
    input_dict = load_dict(path_input_dict)
    target_dict = load_dict(path_target_dict)
    events = pd.read_csv(event_file, header=None).to_numpy()

    for event in events:
        event_str= event[0].split("/")[-1][:-4]
        test_keys= [key for key in input_dict if key.startswith(event_str)]

        # os.makedirs(f"{output_npy_dir}/{EVENT_STR}", exist_ok=True)

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
        
        
        
        data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        output_array, baseline_average_iou = test_model(model, test_loader=data_loader, device=device)
        # test_image_output_array_event_wise(model, path_input_dict, path_target_dict, event_file,batch_size=64, device='cuda',EVENT_STR=EVENT_STR, device_ids=device_ids)






        # Perform LOFO analysis
        feature_importances = []
        for feature in range(test_input_array.shape[1]):
            print(f"\nEvaluating model performance without feature {feature + 1}...")

            # Create a copy of the input data and set the current feature to zero
            modified_data = test_input_array.clone()
            modified_data[:, feature, :, :] = 0

            # Create a new data loader with modified data
            modified_dataset = TensorDataset(modified_data, test_target_array)
            modified_loader = DataLoader(modified_dataset, batch_size=64, shuffle=False)

            # Test model with the modified dataset
            output_array,lofo_iou = test_model(model, test_loader=modified_loader,device=device)
            print(f"mIoU without feature {feature + 1}: {lofo_iou}")

            # Calculate importance as the drop in mIoU
            importance = baseline_average_iou - lofo_iou
            feature_importances.append(importance)

        # Display feature importances
        print("\nFeature importances (based on mIoU drop):")
        for i, imp in enumerate(feature_importances):
            print(f"Feature {i + 1}: Importance = {imp:.4f}")
