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


#########BEFORE#########
# # EVENT_STRS = ["Harvey_20170829_D734_non_flood","Harvey_20170831_9366_non_flood" , "Harvey_20170831_0776_non_flood", "Harvey_20170829_B8C4_non_flood", "Harvey_20170829_3220_non_flood"]
# EVENT_STRS =  ["Harvey_20170829_3220_non_flood"] #["Florence_20180919_374E_non_flood", "Florence_20180919_B86C_non_flood"]
# # Harvey_20170829_3220_non_flood.csv #Harvey_20170829_B8C4_non_flood.csv  #Harvey_20170831_0776_non_flood.csv #Harvey_20170831_9366_non_flood.csv #Harvey_20170829_D734_non_flood.csv
# #"Mississippi_20190617_B9D7_non_flood" #"Mississippi_20190617_B9D7_non_flood"  # "Mississippi_20190617_42BF_non_flood"  #"Mississippi_20190617_3AC6_non_flood" #"Mississippi_20190617_9D85_non_flood" #"Mississippi_20190617_5E5F_non_flood"
# for EVENT_STR in EVENT_STRS:
#     TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_{EVENT_STR}" #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/flood_img_list_flood_non_flood_test2.csv')  #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_3AC6.csv #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_C310.csv  #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_B9D7.csv

#########BEFORE#########



events_file = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv' #combined.csv'
combined_df = pd.read_csv(events_file, header=None) 
for idx, raster_path in enumerate(combined_df[0]): #events = [raster_path.split("/")[-1][:-4] 
    event = raster_path.split("/")[-1][:-4]
    print(event)
    # print(event)
    TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_waterbody/{event}"#f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{event}"
    # raster_paths = glob.glob(f"{TARGET_RASTER_DIR}/*")






# EVENT_STR = "Mississippi_20190617_C310_non_flood" #"Mississippi_20190617_B9D7_non_flood"  # "Mississippi_20190617_42BF_non_flood"  #"Mississippi_20190617_3AC6_non_flood" #"Mississippi_20190617_9D85_non_flood" #"Mississippi_20190617_5E5F_non_flood"
# TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_{EVENT_STR}" #'/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood'  # Reference raster for extent and resolution





    # Input files
    cropped_outpur_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_SM/{event}"  # Folder to save the cropped DEMs  #Mississippi_20190617_5E5F_cropped_dems_non_flood
    
    if os.path.exists(cropped_outpur_dir):
        print('Exists ',cropped_outpur_dir)
        continue
    else:
        print('New ',cropped_outpur_dir)
        os.makedirs(cropped_outpur_dir,exist_ok=True)
        geo_files = glob.glob(f"{TARGET_RASTER_DIR}/*.tif")  #/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_9D85_non_flood/



        # REFERENCE_RASTER_FILES = glob.glob("/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_9D85_non_flood/*.tif") # /p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_Flood_Images_No_Threshold_test #/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_Flood_Images_No_Threshold_test_Mississippi_20190617_3AC6
        # #
        # # /p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood

        # OUTPUT_GEOTIFF_DIR = '/p/lustre2/lazin1/cropped_LC_Mississippi_20190617_9D85_non_flood' #Mississippi_20190617_5E5F_non_flood' #Harvey_D374'
        

        # reference_raster_list = os.listdir(reference_raster_dir)

        # Get reference raster info (extent and resolution)
        temp_raster_path = f"/p/lustre2/lazin1/SMAP_HydroBlocks/SMAP-HydroBlocks_postprocessing/SMAPHB_sample/SMAP-HB_surface-soil-moisture_30m_daily_netcdf/{event}.tif" #/p/lustre2/lazin1/SMAP_HydroBlocks/SMAP-HydroBlocks_postprocessing/SMAPHB_sample/SMAP-HB_surface-soil-moisture_30m_daily_netcdf/non_flood_WM4_S1A_IW_GRDH_1SDV_20170924T001005_20170924T001030_018510_01F303_F9CE.tif
        print(temp_raster_path)
        for reference_raster_file in geo_files:
            f = reference_raster_file.split("/")[-1]
            
            # temp_raster_path = os.path.join(output_geotiff_dir, reference_raster_file.split("/")[-1].split("crop")[0][:-1] + '.tif')
            output_geotiff = os.path.join(cropped_outpur_dir, 'SM_' + reference_raster_file.split("/")[-1])
            
            
            # temp_raster_path = os.path.join(output_geotiff_dir, reference_raster_file[1].split("crop")[0][:-1]+'.tif')
            # output_geotiff = os.path.join(output_geotiff_dir, '1day_prec_' + reference_raster_file[1])
            reference_raster = reference_raster_file #os.path.join(reference_raster_dir,reference_raster_file[1])
            with rasterio.open(reference_raster) as ref:
                ref_transform = ref.transform
                ref_crs = ref.crs
                ref_shape = (ref.height, ref.width)
                ref_bounds = ref.bounds

            # with rasterio.open(
            #     temp_raster_path,
            #     "w",
            #     driver="GTiff",
            #     height=precip_1day.shape[0],
            #     width=precip_1day.shape[1],
            #     count=1,
            #     dtype="float32",
            #     crs="EPSG:4326",
            #     transform=from_origin(longitudes.min(), latitudes.max(), longitudes[1] - longitudes[0], latitudes[1] - latitudes[0])
            # ) as temp_dst:
            #     temp_dst.write((np.flip(precip_1day, axis=0)), 1)

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
