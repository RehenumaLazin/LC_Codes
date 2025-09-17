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




events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_events_short.csv" #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv' #combined.csv'
TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_Sonoma"
combined_df = pd.read_csv(events_file) 
start_dates = combined_df['start'].to_numpy()
end_dates = combined_df['end'].to_numpy() 

events_all = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_events.csv" 
combined_df_all = pd.read_csv(events_all) 
start_dates_all = combined_df_all['start'].to_numpy()
end_dates_all = combined_df_all['end'].to_numpy() 
for idx, (s,e)  in enumerate(zip(start_dates_all, end_dates_all)): #events = [raster_path.split("/")[-1][:-4] 
    print(s,e)
    if idx not in [1, ]: continue
    print(idx, s,e)
    # start_time = datetime.strptime(start_dates[idx] + " 00", "%Y-%m-%d %H")
    # end_time = datetime.strptime(end_dates[idx] + " 23", "%Y-%m-%d %H")
    start_time = datetime.strptime( "2019-02-26 23", "%Y-%m-%d %H")
    end_time = datetime.strptime("2019-02-27 23", "%Y-%m-%d %H")
    
    
    
    # start_dt = datetime.strptime(s + " 00", "%Y-%m-%d %H")
    start_dt = datetime.strptime("2019-02-26 23", "%Y-%m-%d %H")
    # start_date = start_dt.strftime( "%Y-%m-%d %H")
    
    # end_dt = datetime.strptime(e, "%Y-%m-%d")
    end_dt = datetime.strptime("2019-02-27 23", "%Y-%m-%d %H")
    # end_date = end_dt.strftime( "%Y-%m-%d %H")
    
    days = (end_dt - start_dt).days
    
    date_obj = datetime.strptime(s, "%Y-%m-%d")

# Input files
    event_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/event{idx+1}"  # Folder to save the cropped DEMs  #Mississippi_20190617_5E5F_cropped_dems_non_flood

    os.makedirs(event_dir,exist_ok=True)
    geo_files = glob.glob(f"{TARGET_RASTER_DIR}/*.tif")  #/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_9D85_non_flood/


    # Get reference raster info (extent and resolution)
    temp_raster_dir = f"/p/lustre1/lazin1/flood/Sonoma/RussianRiver_{s}_{e}" #/p/lustre2/lazin1/SMAP_HydroBlocks/SMAP-HydroBlocks_postprocessing/SMAPHB_sample/SMAP-HB_surface-soil-moisture_30m_daily_netcdf/non_flood_WM4_S1A_IW_GRDH_1SDV_20170924T001005_20170924T001030_018510_01F303_F9CE.tif
    print(temp_raster_dir)
    if os.path.exists(temp_raster_dir):
        # num_files = len([f for f in os.listdir(temp_raster_dir) if os.path.isfile(os.path.join(temp_raster_dir, f))])
        files = sorted(glob.glob(os.path.join(temp_raster_dir, "depth-1-*.tif")), key=os.path.getmtime)
    
     #[f"{temp_raster_dir}/depth-1-{i * 1440}.tif" for i in range(1, days + 1)] # Get the list of files at 24 th hour interval (24*60)
        for d, file in enumerate(files):

            # # Get the next day
            # next_day = (date_obj + timedelta(days=d+1)).strftime("%Y-%m-%d")
            # os.makedirs(f"{event_dir}/{next_day}",exist_ok=True)
            
            print(file)
            try:
                minutes = int(os.path.basename(file).split("-")[-1].split(".")[0])
                print(minutes)
            except ValueError:
                continue

            timestamp = start_time + timedelta(minutes=minutes)
            print()
            if timestamp <= end_time:
            
                date = timestamp.strftime('%Y%m%d%H')
                os.makedirs(f"{event_dir}/{date}",exist_ok=True)
                
                
                
                for reference_raster_file in geo_files:
                    f = reference_raster_file.split("/")[-1]
                    
                    # temp_raster_path = os.path.join(output_geotiff_dir, reference_raster_file.split("/")[-1].split("crop")[0][:-1] + '.tif')

                    
                    output_geotiff = f"{event_dir}/event{idx+1}_{date}_{f}"
                    print(output_geotiff)
                    if os.path.exists(output_geotiff):
                        print(f"File {output_geotiff} already exists. Skipping...")
                        continue
                    
                    
                    # temp_raster_path = os.path.join(output_geotiff_dir, reference_raster_file[1].split("crop")[0][:-1]+'.tif')
                    # output_geotiff = os.path.join(output_geotiff_dir, '1day_prec_' + reference_raster_file[1])
                    reference_raster = reference_raster_file #os.path.join(reference_raster_dir,reference_raster_file[1])
                    with rasterio.open(reference_raster) as ref:
                        ref_transform = ref.transform
                        ref_crs = ref.crs
                        ref_shape = (ref.height, ref.width)
                        ref_bounds = ref.bounds



                    # Reproject and resample to match the reference raster
                    with rasterio.open(file) as src:
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
