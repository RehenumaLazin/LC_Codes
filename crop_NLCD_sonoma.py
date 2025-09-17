import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio import warp
# from netCDF4 import Dataset
from osgeo import gdal
from datetime import datetime, timedelta
import pandas as pd
import glob




TARGET_RASTER_DIR = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma/"#f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{event}"
# raster_paths = glob.glob(f"{TARGET_RASTER_DIR}/*")






# EVENT_STR = "Mississippi_20190617_C310_non_flood" #"Mississippi_20190617_B9D7_non_flood"  # "Mississippi_20190617_42BF_non_flood"  #"Mississippi_20190617_3AC6_non_flood" #"Mississippi_20190617_9D85_non_flood" #"Mississippi_20190617_5E5F_non_flood"
# TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_{EVENT_STR}" #'/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood'  # Reference raster for extent and resolution





# Input files
cropped_outpur_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_LC/Sonoma/"  # Folder to save the cropped DEMs  #Mississippi_20190617_5E5F_cropped_dems_non_flood
# if not os.path.exists(cropped_outpur_dir):
    
os.makedirs(cropped_outpur_dir,exist_ok=True)
geo_files = sorted(glob.glob(f"{TARGET_RASTER_DIR}/*.tif"))  #/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_9D85_non_flood/



temp_raster_path = f"/p/lustre2/lazin1/Annual_NLCD_LndCov_2023_CU_C1V0.tif"
date_format = "%Y%m%d"
# Get reference raster info (extent and resolution)
for reference_raster_file in geo_files:
    
    # temp_raster_path = os.path.join(output_geotiff_dir, reference_raster_file.split("/")[-1].split("crop")[0][:-1] + '.tif')
    output_geotiff = os.path.join(cropped_outpur_dir, 'LC_' + reference_raster_file.split("/")[-1])
    
    
    # temp_raster_path = os.path.join(output_geotiff_dir, reference_raster_file[1].split("crop")[0][:-1]+'.tif')
    # output_geotiff = os.path.join(output_geotiff_dir, '1day_prec_' + reference_raster_file[1])
    reference_raster = reference_raster_file #os.path.join(reference_raster_dir,reference_raster_file[1])
    with rasterio.open(reference_raster) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)
        ref_bounds = ref.bounds


    # Reproject and resample to match the reference raster
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
