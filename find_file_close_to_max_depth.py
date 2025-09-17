

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Path to the folder containing the maps
folder_path = "/p/vast1/lazin1/triton/output_corrected_event12/asc"
max_file = "MH_1584_00.tif"

# Load the maximum flood depth map
max_depth_map_path = os.path.join(folder_path, max_file)  # Update filename as needed
with rasterio.open(max_depth_map_path) as src:
    max_depth_map = src.read(1)  # Read the first band
    max_profile = src.profile  # Keep profile for metadata

# Normalize and handle NaNs
max_depth_map = np.nan_to_num(max_depth_map)

best_match = None
best_score = float("inf")  # Lower MSE is better
best_map_name = ""

# Function to read and resample a GeoTIFF map
def read_resample_tiff(file_path, reference_profile):
    with rasterio.open(file_path) as src:
        data = src.read(1, out_shape=(reference_profile['height'], reference_profile['width']),
                        resampling=Resampling.bilinear)
        return np.nan_to_num(data)  # Replace NaNs with zeros

# Function to calculate Mean Squared Error (MSE)
def compute_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

# Loop through all flood maps
for file in os.listdir(folder_path):
    if file.endswith(".tif") and file != max_file:  # Ignore the max depth map itself
        flood_map_path = os.path.join(folder_path, file)

        # Read and resample the flood map
        flood_map = read_resample_tiff(flood_map_path, max_profile)

        # Compute Mean Squared Error (MSE)
        mse_score = compute_mse(max_depth_map, flood_map)

        # Update best match
        if mse_score < best_score:  # Lower MSE means better match
            best_score = mse_score
            best_match = flood_map
            best_map_name = file

print(f"The flood map closest to the maximum depth map is: {best_map_name} with MSE score: {best_score:.4f}")
