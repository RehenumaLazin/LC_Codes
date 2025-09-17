import os
import rasterio
import numpy as np
import pandas as pd
import glob

def compute_pixelwise_mean_std(geotiff_files, output_mean_file, output_std_file):
    """
    Compute the mean and standard deviation for each pixel across multiple GeoTIFF files and save as numpy arrays.

    Args:
        geotiff_files (list of str): List of GeoTIFF file paths.
        output_mean_file (str): Path to save the mean numpy array.
        output_std_file (str): Path to save the standard deviation numpy array.

    Returns:
        None
    """
    # Initialize variables for incremental computation
    mean_accumulator = None
    squared_sum_accumulator = None
    count_accumulator = None

    for idx, geotiff_path in enumerate(geotiff_files):
        try:
            # Open GeoTIFF and read the array
            with rasterio.open(geotiff_path) as src:
                array = src.read(1)  # Read the first band
                array = array.astype(np.float64)  # Ensure precision for computation

                # Initialize accumulators on the first iteration
                if mean_accumulator is None:
                    mean_accumulator = np.zeros_like(array, dtype=np.float64)
                    squared_sum_accumulator = np.zeros_like(array, dtype=np.float64)
                    count_accumulator = np.zeros_like(array, dtype=np.int64)

                # Update accumulators (ignore NaN values)
                valid_mask = ~np.isnan(array)
                mean_accumulator[valid_mask] += array[valid_mask]
                squared_sum_accumulator[valid_mask] += array[valid_mask] ** 2
                count_accumulator[valid_mask] += 1

            # Print progress every 1,000 files
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(geotiff_files)} files...")

        except Exception as e:
            print(f"Error processing {geotiff_path}: {e}")

    # Compute mean and standard deviation
    pixelwise_mean = mean_accumulator / count_accumulator
    pixelwise_variance = (squared_sum_accumulator / count_accumulator) - (pixelwise_mean ** 2)
    pixelwise_std_dev = np.sqrt(pixelwise_variance)

    # Save mean and standard deviation as numpy arrays
    np.save(output_mean_file, pixelwise_mean)
    np.save(output_std_file, pixelwise_std_dev)

    print(f"Pixelwise mean saved to {output_mean_file}")
    print(f"Pixelwise standard deviation saved to {output_std_file}")


    # events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Tile_Sonoma.csv" #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined_{EVENT_STR}_No_Threshold.csv" #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined.csv"  #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'

    # tiles_df = pd.read_csv(events_file)
    # tile_names = tiles_df['ras_name'].tolist()  # Assuming the column is named 'tile_name'
TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_Sonoma/" 
tile_names = sorted(glob.glob(f"{TARGET_RASTER_DIR}/*.tif"))


DEM = []
LC = []
day1_prec = []
day5_prec = []
streamflow = []
SM = []
label_geotiff_path = []
for tile_name in tile_names:
    tile_str = tile_name.split("case")[-1]#tile_name.split("crop")[0][:-1] 
    event = tile_name.split("/")[-1].split("_")[0]
    date = tile_name.split("/")[-1].split("_")[1]
    
    DEM.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma/case{tile_str}")
    LC.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_LC/Sonoma/LC_case{tile_str}")
    day1_prec.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/1D_prec/Sonoma/{event}/1day_prec_{tile_name}")
    day5_prec.append( f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/5D_prec/Sonoma/{event}/5day_prec_{tile_name}")
    streamflow.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/streamflow/Sonoma/{event}/streamflow_{tile_name}")
    SM.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_SM/Sonoma/{event}/SM_{tile_name}")
    
    label_geotiff_path.append(f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma/{event}/{tile_name}") 
    
    
# GeoTIFF paths

input_geotiff_paths =[
DEM,
LC,
day1_prec,
day5_prec,
streamflow,
SM
]


# Example usage
geotiff_files = DEM  # Replace with actual paths
output_mean_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/DEM_Sonoma_mean.npy"
output_std_file = f"/p/vast1/lazin1/UNet_inputs/mean_stds/DEM_Sonoma_std_dev.npy"

compute_pixelwise_mean_std(geotiff_files, output_mean_file, output_std_file)
