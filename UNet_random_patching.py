import torch
from torch.utils.data import Dataset
import numpy as np
import random
import rasterio

class FloodPatchDataset(Dataset):
    def __init__(self, input_paths, label_paths, patch_size=128, samples_per_epoch=1000, flood_threshold=0.1, balance_classes=False):
        """
        input_paths: list of input GeoTIFF paths [DEM, prec_day1, prec_day5, etc.]
        label_paths: list of label GeoTIFF paths [flood depth rasters]
        patch_size: int, size of square patch to sample
        samples_per_epoch: int, how many patches you want per epoch
        flood_threshold: float, threshold flood depth (meters) to call a patch flooded
        balance_classes: bool, if True balance flooded/dry patches
        """
        self.input_paths = input_paths
        self.label_paths = label_paths
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.flood_threshold = flood_threshold
        self.balance_classes = balance_classes

        # Open rasters once
        self.inputs_rasters = [rasterio.open(path) for path in input_paths]
        self.labels_rasters = [rasterio.open(path) for path in label_paths]

        # Assume all rasters same size
        self.height, self.width = self.inputs_rasters[0].shape

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        flood_patch = None
        tries = 0

        while flood_patch is None:
            x = random.randint(0, self.width - self.patch_size)
            y = random.randint(0, self.height - self.patch_size)

            # Read label patch (flood depth)
            label_patch = []
            for label_raster in self.labels_rasters:
                label = label_raster.read(1, window=rasterio.windows.Window(x, y, self.patch_size, self.patch_size))
                label_patch.append(label)
            label_patch = np.stack(label_patch, axis=0)[0]  # Assuming 1 label band

            mean_flood = label_patch.mean()

            if not self.balance_classes:
                flood_patch = label_patch
            else:
                # Sample flooded patches or dry patches equally
                if idx % 2 == 0:  # Even indices → flooded
                    if mean_flood >= self.flood_threshold:
                        flood_patch = label_patch
                else:  # Odd indices → dry
                    if mean_flood < self.flood_threshold:
                        flood_patch = label_patch

            tries += 1
            if tries > 10:
                # After 10 tries, just accept whatever patch (avoid infinite loop)
                flood_patch = label_patch

        # Read input patch
        input_patch = []
        for input_raster in self.inputs_rasters:
            inp = input_raster.read(1, window=rasterio.windows.Window(x, y, self.patch_size, self.patch_size))
            input_patch.append(inp)
        input_patch = np.stack(input_patch, axis=0)  # Shape (C, H, W)

        # Normalization (you can adjust or move this outside if you prefer)
        input_patch = (input_patch - np.nanmean(input_patch)) / (np.nanstd(input_patch) + 1e-6)

        return torch.tensor(input_patch, dtype=torch.float32), torch.tensor(flood_patch, dtype=torch.float32).unsqueeze(0)
    
    
    
    input_paths = [
    'DEM.tif',
    'precip_day1.tif',
    'precip_day5.tif',
    'flow.tif',
    'soil_moisture.tif'
]

label_paths = [
    'flood_depth.tif'
]

train_dataset = FloodPatchDataset(
    input_paths=input_paths,
    label_paths=label_paths,
    patch_size=128,
    samples_per_epoch=2000,
    flood_threshold=0.1,  # 10cm
    balance_classes=True  # balance wet/dry
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

