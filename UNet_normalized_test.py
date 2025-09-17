import torch
import rasterio
import numpy as np
from torch.utils.data import DataLoader
from rasterio.transform import from_origin

# Define the FloodDataset class if not already defined
class FloodDataset:
    def __init__(self, geotiff_paths, labels_paths=None):
        """
        Custom Dataset for Flood Segmentation.

        Args:
            geotiff_paths (list of str): List of paths to input GeoTIFF files.
            labels_paths (list of str, optional): List of paths to label GeoTIFF files (for training).
        """
        self.geotiff_paths = geotiff_paths
        self.labels_paths = labels_paths

    def __len__(self):
        return len(self.geotiff_paths)

    def __getitem__(self, idx):
        # Read GeoTIFF files for inputs
        with rasterio.open(self.geotiff_paths[idx]) as src:
            inputs = src.read()  # Shape: (channels, H, W)
            transform = src.transform
            crs = src.crs

        # Normalize inputs
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        inputs = torch.tensor(inputs, dtype=torch.float32)

        if self.labels_paths:
            # Read label GeoTIFF file
            with rasterio.open(self.labels_paths[idx]) as src:
                label = src.read(1)  # Shape: (H, W)
            label = (label > 0).astype(np.float32)
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
            return inputs, label, transform, crs

        return inputs, transform, crs


def load_model(model_path, device):
    """
    Load the trained U-Net model.

    Args:
        model_path (str): Path to the saved model (.pt file).
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        model: The loaded U-Net model.
    """
    model = UNet(in_channels=6, out_channels=1, base_channels=32)  # Adjust according to your architecture
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(dataloader):
        inputs, transform, crs = data[0].to(device), data[1][0], data[2][0]
        with torch.no_grad():
            outputs = model(inputs.unsqueeze(0))  # Add batch dimension
            outputs = torch.sigmoid(outputs).squeeze().cpu().numpy()

        # Binarize the output (optional)
        output_binary = (outputs > 0.5).astype(np.float32)

        # Save the result as a GeoTIFF
        with rasterio.open(
            output_geotiff_paths[i],
            "w",
            driver="GTiff",
            height=output_binary.shape[0],
            width=output_binary.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(output_binary, 1)

        print(f"Output saved as GeoTIFF: {output_geotiff_paths[i]}")


# Example Usage
input_image_paths = ["path/to/input_image_1.tif", "path/to/input_image_2.tif"]  # Replace with your file paths
output_geotiff_paths = ["output_result_1.tif", "output_result_2.tif"]  # Replace with desired output paths

# Load dataset
dataset = FloodDataset(input_image_paths)

# Load the trained model
model_path = "unet_model.pt"  # Path to the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(model_path, device)

# Test the model and save results
test_unet_on_dataset(model, dataset, output_geotiff_paths, device)
