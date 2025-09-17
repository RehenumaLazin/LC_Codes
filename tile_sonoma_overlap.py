import os
from osgeo import gdal
import numpy as np

def crop_geotiff(input_tiff_path, output_dir, crop_width, crop_height, overlap=16):
    # Open the large GeoTIFF image
    ds = gdal.Open(input_tiff_path)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    No_data = band.GetNoDataValue()
    arr = np.array(band.ReadAsArray())
    
    img_width = ds.RasterXSize
    img_height = ds.RasterYSize

    # Calculate step size (ensuring overlap)
    step_x = crop_width - overlap
    step_y = crop_height - overlap

    count = 1
    for offset_x in range(0, img_width - overlap, step_x):
        for offset_y in range(0, img_height - overlap, step_y):
            # Ensure we don't exceed image boundaries
            if offset_x + crop_width > img_width:
                offset_x = img_width - crop_width
            if offset_y + crop_height > img_height:
                offset_y = img_height - crop_height
            
            # Read the data for the tile
            tile_data = band.ReadAsArray(offset_x, offset_y, crop_width, crop_height)
            tile_data = np.where(tile_data == No_data, np.nan, tile_data)
            
            # Skip tile if all values are NaN
            if np.isnan(tile_data).any():
                continue
            
            # Create output file
            driver = gdal.GetDriverByName('GTiff')
            output_file = os.path.join(output_dir,  f'Sonoma_crop_{count}.tif')
            out_ds = driver.Create(output_file, crop_width, crop_height, 1, band.DataType)
            
            # Set GeoTransform and Projection for the cropped file
            new_gt = (
                gt[0] + offset_x * gt[1], gt[1], gt[2],
                gt[3] + offset_y * gt[5], gt[4], gt[5]
            )
            out_ds.SetGeoTransform(new_gt)
            out_ds.SetProjection(ds.GetProjection())
            
            # Write cropped data
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(tile_data)
            out_band.FlushCache()

            # Clean up
            out_ds = None
            count += 1

# Parameters
crop_width = 512
crop_height = 512
overlap = 256

# Input and output paths
raster_path = "/p/lustre2/lazin1/DEM/cropped_dem_11467270_extent.tif"
output_dir = "/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_Sonoma"
os.makedirs(output_dir, exist_ok=True)

# Process the image
try:
    print(f"Processing {raster_path}...")
    crop_geotiff(raster_path, output_dir, crop_width, crop_height, overlap)
    print("Cropping complete.")
except Exception as e:
    print(f"Error: {e}")
