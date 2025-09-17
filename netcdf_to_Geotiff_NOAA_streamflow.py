import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio import warp
from netCDF4 import Dataset
from osgeo import gdal
from datetime import datetime, timedelta
import pandas as pd
import glob
import calendar

import xarray as xr
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata




############ BEFORE#################

events_file = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'
combined_df = pd.read_csv(events_file, header=None) 
for idx, raster_path in enumerate(combined_df[0]): #events = [raster_path.split("/")[-1][:-4] 
    event = raster_path.split("/")[-1][:-4]
    # print(event)
    TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{event}"#f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_waterbody/{event}"#f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{event}"

# EVENT_STR = "Mississippi_20190617_C310_non_flood" #"Mississippi_20190617_B9D7_non_flood"  # "Mississippi_20190617_42BF_non_flood"  #"Mississippi_20190617_3AC6_non_flood" #"Mississippi_20190617_9D85_non_flood" #"Mississippi_20190617_5E5F_non_flood"
# TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_{EVENT_STR}" #'/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood'  # Reference raster for extent and resolution

    # reference_raster_dir = '/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_Flood_Images'  # Reference raster for extent and resolution

    # reference_raster_files = pd.read_csv('/usr/workspace/lazin1/anaconda_dane/envs/RAPID/selected_raster_list.csv')
    reference_raster_files = glob.glob(f"{TARGET_RASTER_DIR}/*.tif") #'/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood'
    # reference_raster_files= reference_raster_files.to_numpy()


    output_geotiff_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/streamflow/{event}" #streamflow_No_Waterbody  #5E5F #/p/vast1/lazin1/UNet_inputs/Geotiff_var/streamflow_No_Waterbody
    if os.path.exists(output_geotiff_dir):
        print('exist', output_geotiff_dir)
        continue
    else:
        print('new dir', output_geotiff_dir)
        os.makedirs(output_geotiff_dir,exist_ok=True)


        date_format = "%Y%m%d"
        # Get reference raster info (extent and resolution)
        for reference_raster_file in reference_raster_files:
            precip_1day = []
            
            temp_raster_path = os.path.join(output_geotiff_dir, reference_raster_file.split("/")[-1].split("crop")[0][:-1] + '.tif')
            output_geotiff = os.path.join(output_geotiff_dir, 'streamflow_'+ reference_raster_file.split("/")[-1])
            
            
            # temp_raster_path = os.path.join(output_geotiff_dir, reference_raster_file[1].split("crop")[0][:-1]+'.tif')
            # output_geotiff = os.path.join(output_geotiff_dir, '1day_prec_' + reference_raster_file[1])
            reference_raster = reference_raster_file #os.path.join(reference_raster_dir,reference_raster_file[1])
            with rasterio.open(reference_raster) as ref:
                ref_transform = ref.transform
                ref_crs = ref.crs
                ref_shape = (ref.height, ref.width)
                ref_bounds = ref.bounds

            if not os.path.exists(temp_raster_path):
            
                print(reference_raster_file)
                
                
                date_str = reference_raster_file.split("/")[-1].split("crop")[0][-51:-43]  #LAZIN120412245
                reference_raster_file[2][:-7]
                # end_date_dt = datetime.fromtimestamp(datetime.strptime(end_date_str, "%Y%m%d").timestamp()) + timedelta(days=1)
                # date_dt = datetime.fromtimestamp(datetime.strptime(date_str +" 23" , "%Y%m%d %H").timestamp()) #datetime.strptime(end_date_str+" 23", "%Y%m%d %H").timestamp()
                y = date_str[:4]
                m = date_str[4:6]
                day = date_str[6:8]
                # date = datetime.strptime(date_str, "%Y%m%d")
                # date = datetime(y, m, d).strftime("%Y-%m-%d")
                date = f"{y}{m}{day}"
                
                file_path = f"/p/lustre2/lazin1/NOAA_streamflow_WRF_Hydro/{y}/{date}.nc"

                dataset = xr.open_dataset(file_path)

                # Step 2: Explore the dataset and extract variables
                # print(dataset)

                # Extract streamflow, latitude, and longitude
                streamflow = dataset['streamflow'].values  # 1D streamflow data #qSfcLatRunoff
                lat = dataset['latitude'].values  # 1D latitude
                lon = dataset['longitude'].values  # 1D longitude

                # Step 1: Example 1D scatter data (replace with your own lat, lon, and streamflow values)
                # lat = np.array([40.1, 40.5, 40.8, 41.0, 41.3])  # Replace with your latitudes
                # lon = np.array([-105.3, -105.0, -104.7, -104.4, -104.1])  # Replace with your longitudes
                # streamflow = np.array([12.5, 18.3, 15.7, 10.4, 25.1])  # Replace with your streamflow values

                # Step 2: Define grid resolution and extent
                grid_resolution = 0.0009  # Grid cell size (in degrees)
                grid_lon, grid_lat = np.meshgrid(
                    np.arange(lon.min(), lon.max(), grid_resolution),
                    np.arange(lat.min(), lat.max(), grid_resolution)
                )

                # Step 3: Interpolate the scatter points onto the 2D grid
                grid_streamflow = griddata(
                    (lon, lat),  # Points (1D arrays)
                    streamflow,  # Values at the points
                    (grid_lon, grid_lat),  # Target grid
                    method='linear'  # Linear interpolation
                )

                # Step 4: Define transformation and CRS (WGS84, EPSG:4326)
                transform = from_origin(grid_lon.min(), grid_lat.max(), grid_resolution, grid_resolution)

                # Step 5: Save the 2D grid as a GeoTIFF file using rasterio
                output_tiff = temp_raster_path#f"{output_geotiff_dir}/scatter_streamflow.tif"
                with rasterio.open(
                    output_tiff,
                    'w',
                    driver='GTiff',
                    height=grid_streamflow.shape[0],
                    width=grid_streamflow.shape[1],
                    count=1,
                    dtype=grid_streamflow.dtype,
                    crs='EPSG:4326',  # WGS84
                    transform=transform,
                ) as dst:
                    dst.write(np.nan_to_num(np.flip(grid_streamflow, axis=0)), 1)  # Fill NaNs with zeros or appropriate values

                print(f"\nGeoTIFF saved as: {output_tiff}")
                
                with rasterio.open(temp_raster_path) as src:
                    with rasterio.open(
                        output_geotiff,
                        "w",
                        driver="GTiff",
                        height=ref_shape[0],
                        width=ref_shape[1],
                        count=1,
                        dtype="float32",
                        crs=ref_crs,
                        transform=ref_transform,
                        nodata=-9999
                    ) as dst:
                        # Resample and reproject the data
                        warp.reproject(
                            source=rasterio.band(src, 1),
                            destination=rasterio.band(dst, 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=warp.Resampling.nearest
                        )

                print(f"Streamflow GeoTIFF saved at {output_geotiff}")

                
            else:
                with rasterio.open(temp_raster_path) as src:
                    with rasterio.open(
                        output_geotiff,
                        "w",
                        driver="GTiff",
                        height=ref_shape[0],
                        width=ref_shape[1],
                        count=1,
                        dtype="float32",
                        crs=ref_crs,
                        transform=ref_transform,
                        nodata=-9999
                    ) as dst:
                        # Resample and reproject the data
                        warp.reproject(
                            source=rasterio.band(src, 1),
                            destination=rasterio.band(dst, 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=warp.Resampling.nearest
                        )

            print(f"Streamflow GeoTIFF saved at {output_geotiff}")
