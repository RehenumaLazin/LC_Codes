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


TARGET_RASTER_DIR = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/Sonoma/"#f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{event}"reference_raster_files = sorted(glob.glob(f"{TARGET_RASTER_DIR}/*.tif")) #'/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood'







file_info= f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/geotiff_info_for_SM_Sonoma.csv" # geotiff_info_for_SM_Sonoma_event3.csv #geotiff_info_for_SM.csv
info_df = pd.read_csv(file_info)
filenames = info_df['filename'].tolist() 

# Input files
cropped_outpur_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_SM/Sonoma/event2/"  # Folder to save the cropped DEMs  #Mississippi_20190617_5E5F_cropped_dems_non_flood


for SM_file in filenames:
    

    os.makedirs(cropped_outpur_dir,exist_ok=True)
    geo_files = sorted(glob.glob(f"{TARGET_RASTER_DIR}/*.tif"))  #/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_9D85_non_flood/

    # Get reference raster info (extent and resolution)
    temp_raster_path = f"/p/lustre2/lazin1/SMAP_HydroBlocks/SMAP-HydroBlocks_postprocessing/SMAPHB_sample/SMAP-HB_surface-soil-moisture_30m_daily_netcdf/{SM_file}" #/p/lustre2/lazin1/SMAP_HydroBlocks/SMAP-HydroBlocks_postprocessing/SMAPHB_sample/SMAP-HB_surface-soil-moisture_30m_daily_netcdf/non_flood_WM4_S1A_IW_GRDH_1SDV_20170924T001005_20170924T001030_018510_01F303_F9CE.tif
    print(temp_raster_path)
    for reference_raster_file in geo_files:
        f = reference_raster_file.split("/")[-1]
        
        # temp_raster_path = os.path.join(output_geotiff_dir, reference_raster_file.split("/")[-1].split("crop")[0][:-1] + '.tif')
        output_geotiff = os.path.join(cropped_outpur_dir, 'SM_event2_'+ SM_file.replace(".tif","_") + reference_raster_file.split("/")[-1])
        
        
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
