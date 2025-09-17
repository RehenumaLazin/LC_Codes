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

############### BEFORE############
# EVENT_STRS = ["Harvey_20170829_D734_non_flood","Harvey_20170831_9366_non_flood" , "Harvey_20170831_0776_non_flood", "Harvey_20170829_B8C4_non_flood", "Harvey_20170829_3220_non_flood"]
# EVENT_STRS = ["Florence_20180919_374E_non_flood", "Florence_20180919_B86C_non_flood"]
# EVENT_STRS= ["Harvey_20170829_3220_non_flood"]

# # Harvey_20170829_3220_non_flood.csv #Harvey_20170829_B8C4_non_flood.csv  #Harvey_20170831_0776_non_flood.csv #Harvey_20170831_9366_non_flood.csv #Harvey_20170829_D734_non_flood.csv
# #"Mississippi_20190617_B9D7_non_flood" #"Mississippi_20190617_B9D7_non_flood"  # "Mississippi_20190617_42BF_non_flood"  #"Mississippi_20190617_3AC6_non_flood" #"Mississippi_20190617_9D85_non_flood" #"Mississippi_20190617_5E5F_non_flood"
# for EVENT_STR in EVENT_STRS:
#     TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_{EVENT_STR}" #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/flood_img_list_flood_non_flood_test2.csv')  #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_3AC6.csv #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_C310.csv  #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_B9D7.csv
########### BEFORE#############


events_file = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'
combined_df = pd.read_csv(events_file, header=None) 
for idx, raster_path in enumerate(combined_df[0]): #events = [raster_path.split("/")[-1][:-4] 
    event = raster_path.split("/")[-1][:-4]
    # print(event)
    TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_waterbody/{event}"#f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{event}"

# EVENT_STR = "Mississippi_20190617_C310_non_flood" #"Mississippi_20190617_B9D7_non_flood"  # "Mississippi_20190617_42BF_non_flood"  #"Mississippi_20190617_3AC6_non_flood" #"Mississippi_20190617_9D85_non_flood" #"Mississippi_20190617_5E5F_non_flood"
# TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_{EVENT_STR}" #'/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood'  # Reference raster for extent and resolution

    # reference_raster_dir = '/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_Flood_Images'  # Reference raster for extent and resolution

    # reference_raster_files = pd.read_csv('/usr/workspace/lazin1/anaconda_dane/envs/RAPID/selected_raster_list.csv')
    reference_raster_files = glob.glob(f"{TARGET_RASTER_DIR}/*.tif") #'/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood'
    # reference_raster_files= reference_raster_files.to_numpy()


    output_geotiff_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/5D_prec/{event}"  #5E5F
    if os.path.exists(output_geotiff_dir):
        print('exist', output_geotiff_dir)
        continue
    else:
        print('new dir', output_geotiff_dir)
        
        os.makedirs(output_geotiff_dir, exist_ok=True)

        # reference_raster_list = os.listdir(reference_raster_dir)






        date_format = "%Y%m%d"
        # Get reference raster info (extent and resolution)
        for reference_raster_file in reference_raster_files:
            precip_5day = []
            reference_raster = reference_raster_file #os.path.join(reference_raster_dir,reference_raster_file[1])
            with rasterio.open(reference_raster) as ref:
                ref_transform = ref.transform
                ref_crs = ref.crs
                ref_shape = (ref.height, ref.width)
                ref_bounds = ref.bounds
            temp_raster_path = os.path.join(output_geotiff_dir, reference_raster_file.split("/")[-1].split("crop")[0][:-1] + '.tif')
            output_geotiff = os.path.join(output_geotiff_dir, '5day_prec_' + reference_raster_file.split("/")[-1])

            if not os.path.exists(temp_raster_path):
            
                # print(reference_raster_file)
                
                
                end_date_str = reference_raster_file.split("/")[-1].split("crop")[0][-51:-43]  #reference_raster_file[2][:-7]
                # end_date_dt = datetime.fromtimestamp(datetime.strptime(end_date_str, "%Y%m%d").timestamp()) + timedelta(days=1)
                end_date_dt = datetime.fromtimestamp(datetime.strptime(end_date_str +" 23" , "%Y%m%d %H").timestamp()) #datetime.strptime(end_date_str+" 23", "%Y%m%d %H").timestamp()
                y=end_date_str[:4]
                m = end_date_str[4:6]
                day = end_date_str[6:8]
                end_date = datetime.strptime(end_date_str, "%Y%m%d")
                if int(day) < calendar.monthrange(int(y),int(m))[1]:
                    end_date_dt = datetime.fromtimestamp(datetime.strptime(end_date_str +" 23" , "%Y%m%d %H").timestamp())
                else:
                    end_date_dt = datetime.fromtimestamp(datetime.strptime(end_date_str +" 15" , "%Y%m%d %H").timestamp())
                
                
                # end_date_dt = datetime.fromtimestamp(datetime.strptime(end_date_str +" 23" , "%Y%m%d %H").timestamp())
                start_date_dt = datetime.fromtimestamp(datetime.strptime(end_date_str +" 23" , "%Y%m%d %H").timestamp()) - timedelta(days=5) + timedelta(hours=1)
                print(start_date_dt,end_date_dt)
                # time_difference = end_date - datetime(int(y), int(m), 1)

                # # Convert the time difference to hours
                # hours = time_difference.total_seconds() / 3600 + 24
                if start_date_dt.year == end_date_dt.year and start_date_dt.month == end_date_dt.month:
                    input_netcdf = os.path.join('/p/lustre2/lazin1/AORC_APCP_surface/APCP_surface_'+ y + '_' + m+ '.nc') 
                    # Open NetCDF file
                    nc = Dataset(input_netcdf, 'r')
                    precip_var = nc.variables['APCP_surface']  # Replace with actual variable name
                    latitudes = nc.variables['latitude'][:]
                    longitudes = nc.variables['longitude'][:]
                    time_var = nc.variables['time']

                    # Find the start index for the 3-day period
                    # time_units = time_var.units  # Expected format: "hours since YYYY-MM-DD HH:MM:SS"
                    # base_time = datetime.strptime(time_units.split("since ")[1], "%Y-%m-%d %H:%M:%S")
                    end_date_idx = (np.where(time_var[:] == end_date_dt.timestamp())[0][0])
                    start_date_idx = (np.where(time_var[:] == start_date_dt.timestamp())[0][0]) 
                    print(start_date_idx,end_date_idx)

                    # Sum precipitation over the previous 24 hours (1 day)
                    precip_5day = np.sum(precip_var[start_date_idx:end_date_idx+1, :, :], axis=0)
                elif start_date_dt.year == end_date_dt.year and   end_date_dt.month - start_date_dt.month==1:
                    input_netcdf_start = os.path.join('/p/lustre2/lazin1/AORC_APCP_surface/APCP_surface_'+ y + '_' + (str(start_date_dt.month).zfill(2))+ '.nc')
                    input_netcdf_end = os.path.join('/p/lustre2/lazin1/AORC_APCP_surface/APCP_surface_'+ y + '_' + m+ '.nc')  
                    
                    # load start date variables
                    nc_start = Dataset(input_netcdf_start, 'r')
                    precip_var_start = nc_start.variables['APCP_surface']  # Replace with actual variable name
                    latitudes = nc_start.variables['latitude'][:]
                    longitudes = nc_start.variables['longitude'][:]
                    time_var_start = nc_start.variables['time']
                    # get start date idx
                    start_date_idx = (np.where(time_var_start[:] == start_date_dt.timestamp())[0][0]) 
                    
                    
                    # load end date variables
                    nc_end = Dataset(input_netcdf_end, 'r')
                    precip_var_end = nc_end.variables['APCP_surface']  # Replace with actual variable name
                    # latitudes_end = nc_end.variables['latitude'][:]
                    # longitudes_end = nc_end.variables['longitude'][:]
                    time_var_end = nc_end.variables['time']
                    end_date_idx = (np.where(time_var_end[:] == end_date_dt.timestamp())[0][0])
                    
                    
                    print(end_date_idx, start_date_idx)

                    # Sum precipitation over the previous 24 hours (1 day)
                    precip_sum_start = np.sum(precip_var_start[start_date_idx:-1, :, :], axis=0)
                    precip_sum_end = np.sum(precip_var_end[0:end_date_idx+1, :, :], axis=0)
                    
                    precip_5day = np.sum([precip_sum_start, precip_sum_end], axis=0)
                    
                elif start_date_dt.year < end_date_dt.year and start_date_dt.month==12 and end_date_dt.month==1:
                    input_netcdf_start = os.path.join('/p/lustre2/lazin1/AORC_APCP_surface/APCP_surface_'+ str(start_date_dt.year) + '_' + (str(start_date_dt.month).zfill(2))+ '.nc')
                    input_netcdf_end = os.path.join('/p/lustre2/lazin1/AORC_APCP_surface/APCP_surface_'+ y + '_' + m+ '.nc')  
                    
                    # load start date variables
                    nc_start = Dataset(input_netcdf_start, 'r')
                    precip_var_start = nc_start.variables['APCP_surface']  # Replace with actual variable name
                    latitudes = nc_start.variables['latitude'][:]
                    longitudes = nc_start.variables['longitude'][:]
                    time_var_start = nc_start.variables['time']
                    # get start date idx
                    start_date_idx = (np.where(time_var_start[:] == start_date_dt.timestamp())[0][0]) 
                    
                    
                    # load end date variables
                    nc_end = Dataset(input_netcdf_end, 'r')
                    precip_var_end = nc_end.variables['APCP_surface']  # Replace with actual variable name
                    # latitudes_end = nc_end.variables['latitude'][:]
                    # longitudes_end = nc_end.variables['longitude'][:]
                    time_var_end = nc_end.variables['time']
                    end_date_idx = (np.where(time_var_end[:] == end_date_dt.timestamp())[0][0])
                    
                    
                    print(end_date_idx, start_date_idx)

                    # Sum precipitation over the previous 24 hours (1 day)
                    precip_sum_start = np.sum(precip_var_start[start_date_idx:-1, :, :], axis=0)
                    precip_sum_end = np.sum(precip_var_end[0:end_date_idx+1, :, :], axis=0)
                    
                    precip_5day = np.sum([precip_sum_start, precip_sum_end], axis=0)        

                    
                elif start_date_dt.year == end_date_dt.year and start_date_dt.month==1 and end_date_dt.month==3:
                    input_netcdf_start = os.path.join('/p/lustre2/lazin1/AORC_APCP_surface/APCP_surface_'+ str(start_date_dt.year) + '_' + (str(start_date_dt.month).zfill(2))+ '.nc')
                    input_netcdf_middle = os.path.join('/p/lustre2/lazin1/AORC_APCP_surface/APCP_surface_'+ str(start_date_dt.year) + '_' + (str(start_date_dt.month+1).zfill(2))+ '.nc')
                    input_netcdf_end = os.path.join('/p/lustre2/lazin1/AORC_APCP_surface/APCP_surface_'+ y + '_' + m+ '.nc')  
                    
                    # load start date variables
                    nc_start = Dataset(input_netcdf_start, 'r')
                    precip_var_start = nc_start.variables['APCP_surface']  # Replace with actual variable name
                    latitudes = nc_start.variables['latitude'][:]
                    longitudes = nc_start.variables['longitude'][:]
                    time_var_start = nc_start.variables['time']
                    # get start date idx
                    start_date_idx = (np.where(time_var_start[:] == start_date_dt.timestamp())[0][0]) 
                    
                    # load February variables
                    nc_middle = Dataset(input_netcdf_middle, 'r')
                    precip_var_middle = nc_middle.variables['APCP_surface']  # Replace with actual variable name
                    # latitudes_middle = nc_middle.variables['latitude'][:]
                    # longitudes_middle = nc_middle.variables['longitude'][:]
                    time_var_middle = nc_middle.variables['time']
                    
                    
                    # load end date variables
                    nc_end = Dataset(input_netcdf_end, 'r')
                    precip_var_end = nc_end.variables['APCP_surface']  # Replace with actual variable name
                    # latitudes_end = nc_end.variables['latitude'][:]
                    # longitudes_end = nc_end.variables['longitude'][:]
                    time_var_end = nc_end.variables['time']
                    end_date_idx = (np.where(time_var_end[:] == end_date_dt.timestamp())[0][0])
                    
                    
                    print(end_date_idx, start_date_idx)

                    # Sum precipitation over the previous 24 hours (1 day)
                    precip_sum_start = np.sum(precip_var_start[start_date_idx:-1, :, :], axis=0)
                    precip_sum_end = np.sum(precip_var_end[0:end_date_idx+1, :, :], axis=0)
                    
                    precip_5day = np.sum([precip_sum_start, precip_var_middle,precip_sum_end], axis=0)     
                    
                    
                # with rasterio.open(reference_raster) as ref:
                #     ref_transform = ref.transform
                #     ref_crs = ref.crs
                #     ref_shape = (ref.height, ref.width)
                #     ref_bounds = ref.bounds

                # Create a temporary in-memory raster for the aggregated precipitation
                temp_raster_path = os.path.join(output_geotiff_dir, reference_raster_file.split("/")[-1].split("crop")[0][:-1] + '.tif')
                with rasterio.open(
                    temp_raster_path,
                    "w",
                    driver="GTiff",
                    height=precip_5day.shape[0],
                    width=precip_5day.shape[1],
                    count=1,
                    dtype="float32",
                    crs="EPSG:4326",
                    transform=from_origin(longitudes.min(), latitudes.max(), longitudes[1] - longitudes[0], latitudes[1] - latitudes[0])
                ) as temp_dst:
                    temp_dst.write((np.flip(precip_5day, axis=0)), 1)

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

                # Clean up temporary file
                import os
                # os.remove(temp_raster_path)

                # Close the NetCDF dataset
                # nc.close()
                
            else:
                # print('exist',temp_rastecd r_path)
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

            print(f"5-day precipitation GeoTIFF saved at {output_geotiff}")
