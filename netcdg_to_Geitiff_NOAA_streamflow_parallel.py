import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio import warp
from datetime import datetime, timedelta
import pandas as pd
import glob
import xarray as xr
from scipy.interpolate import griddata
from multiprocessing import Pool

def process_event(raster_path):
    for raster_path in raster_paths:
        event = raster_path.split("/")[-1][:-4]
        print(event)
        # TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{event}"
        # output_geotiff_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/streamflow/{event}"
        TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{event}"
        output_geotiff_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/streamflow/{event}"
        if not os.path.exists(output_geotiff_dir):
            os.makedirs(output_geotiff_dir, exist_ok=True)

        reference_raster_files = glob.glob(f"{TARGET_RASTER_DIR}/*.tif")

        for reference_raster_file in reference_raster_files:
            print(reference_raster_file)
            try:
                # with rasterio.open(reference_raster_file) as ref:
                #     ref_transform = ref.transform
                #     ref_crs = ref.crs
                #     ref_shape = (ref.height, ref.width)

                date_str = reference_raster_file.split("/")[-1].split("crop")[0][-51:-43]
                
                y, m, d = date_str[:4], date_str[4:6], date_str[6:8]
                file_path = f"/p/lustre2/lazin1/NOAA_streamflow_WRF_Hydro/{y}/{y}{m}{d}.nc"
                
                dataset = xr.open_dataset(file_path)
                streamflow = dataset['streamflow'].values
                lat, lon = dataset['latitude'].values, dataset['longitude'].values
                grid_resolution = 0.0009
                grid_lon, grid_lat = np.meshgrid(
                    np.arange(lon.min(), lon.max(), grid_resolution),
                    np.arange(lat.min(), lat.max(), grid_resolution)
                )
                grid_streamflow = griddata(
                    (lon, lat),
                    streamflow,
                    (grid_lon, grid_lat),
                    method='linear'
                )
                transform = from_origin(grid_lon.min(), grid_lat.max(), grid_resolution, grid_resolution)
                output_tiff = os.path.join(output_geotiff_dir, f'streamflow_{event}.tif')

                with rasterio.open(
                    output_tiff, 'w', driver='GTiff',
                    height=grid_streamflow.shape[0],
                    width=grid_streamflow.shape[1],
                    count=1, dtype=str(grid_streamflow.dtype),
                    crs='EPSG:4326', transform=transform
                ) as dst:
                    dst.write(np.nan_to_num(np.flip(grid_streamflow, axis=0)), 1)

                print(f"Streamflow GeoTIFF saved at {output_tiff}")
            except Exception as e:
                print(f"Error processing {reference_raster_file}: {e}")

# if __name__ == "__main__":
events_file = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'
combined_df = pd.read_csv(events_file, header=None)
raster_paths = combined_df[0].tolist()
process_event(raster_paths)

# with Pool(processes=os.cpu_count()) as pool:
#     print('PRINT',raster_paths[0].split("/")[-1][:-4])
#     results = pool.map(process_event, raster_paths)

# for result in results:
#     print(result)
