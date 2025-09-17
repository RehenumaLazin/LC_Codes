import os
import subprocess
from osgeo import gdal
import glob
import pandas as pd

########BEFORE##############
# # EVENT_STRS = ["Harvey_20170829_D734_non_flood","Harvey_20170831_9366_non_flood" , "Harvey_20170831_0776_non_flood", "Harvey_20170829_B8C4_non_flood", "Harvey_20170829_3220_non_flood"]
# EVENT_STRS = ["Florence_20180919_374E_non_flood", "Florence_20180919_B86C_non_flood"]
# # Harvey_20170829_3220_non_flood.csv #Harvey_20170829_B8C4_non_flood.csv  #Harvey_20170831_0776_non_flood.csv #Harvey_20170831_9366_non_flood.csv #Harvey_20170829_D734_non_flood.csv
# #"Mississippi_20190617_B9D7_non_flood" #"Mississippi_20190617_B9D7_non_flood"  # "Mississippi_20190617_42BF_non_flood"  #"Mississippi_20190617_3AC6_non_flood" #"Mississippi_20190617_9D85_non_flood" #"Mississippi_20190617_5E5F_non_flood"
# for EVENT_STR in EVENT_STRS:
#     TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_{EVENT_STR}" #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/flood_img_list_flood_non_flood_test2.csv')  #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_3AC6.csv #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_C310.csv  #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_B9D7.csv

########### BEFORE############


events_file = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'
combined_df = pd.read_csv(events_file, header=None) 
for idx, raster_path in enumerate(combined_df[0]): #events = [raster_path.split("/")[-1][:-4] 
    event = raster_path.split("/")[-1][:-4]
    # print(event)
    TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{event}" #f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_waterbody/{event}"#f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{event}"



# EVENT_STR = "Mississippi_20190617_C310_non_flood" #"Mississippi_20190617_B9D7_non_flood"  # "Mississippi_20190617_42BF_non_flood"  #"Mississippi_20190617_3AC6_non_flood" #"Mississippi_20190617_9D85_non_flood" #"Mississippi_20190617_5E5F_non_flood"
# TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_{EVENT_STR}" #'/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood'  # Reference raster for extent and resolution





    # Input files
    cropped_outpur_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/{event}"  # Folder to save the cropped DEMs  #Mississippi_20190617_5E5F_cropped_dems_non_flood
    
    
    if os.path.exists(cropped_outpur_dir):
        # print('Exists ',cropped_outpur_dir)
        continue
    else:
        print('New ',cropped_outpur_dir)
        os.makedirs(cropped_outpur_dir,exist_ok=True)
        # geo_files = glob.glob(f"{TARGET_RASTER_DIR}/*.tif")  #/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_9D85_non_flood/

    
        
        
        geo_files = glob.glob(f"{TARGET_RASTER_DIR}/*.tif")  #/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_9D85_non_flood/
                                                                                                                                        # /p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood


        # Create output folder if it doesn't exist
        os.makedirs(cropped_outpur_dir, exist_ok=True)


        merged_vrt = "/p/lustre1/lazin1/USGS_DEM_10m/merged_USGS_DEM_10m.vrt"  # The merged VRT file
        # watershed_shp = "/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Mississippi_20190617_5E5F_Flood_imgs_shapefile.shp"  # Shapefile with 20 watersheds





        # # Step 1: Merge all DEM tiles into a VRT
        # dem_tiles = [os.path.join(dem_tiles_directory, f) for f in os.listdir(dem_tiles_directory) if f.endswith('.tif')]
        # subprocess.run(["gdalbuildvrt", merged_vrt] + dem_tiles)
        # print(f"Merged VRT created at {merged_vrt}")

        # Step 2: Extract bounding boxes from GeoTIFF files and crop the VRT
        for i, geo_file in enumerate(geo_files, start=1):
            # Open GeoTIFF to get its bounding box
            
            # Step 1: Extract metadata from the low-resolution raster
            low_res_ds = gdal.Open(geo_file)
            low_res_geotransform = low_res_ds.GetGeoTransform()
            low_res_projection = low_res_ds.GetProjection()

            # Extract bounding box, pixel size, and dimensions
            min_x = low_res_geotransform[0]
            max_y = low_res_geotransform[3]
            pixel_width = low_res_geotransform[1]
            pixel_height = low_res_geotransform[5]
            rows = low_res_ds.RasterYSize
            cols = low_res_ds.RasterXSize
            max_x = min_x + cols * pixel_width
            min_y = max_y + rows * pixel_height
            low_res_ds = None  # Close the dataset
            
            # ds = gdal.Open(geo_file)
            # geotransform = ds.GetGeoTransform()
            # min_x = geotransform[0]
            # max_y = geotransform[3]
            # max_x = min_x + geotransform[1] * ds.RasterXSize
            # min_y = max_y + geotransform[5] * ds.RasterYSize
            # ds = None  # Close the dataset

            # Define output cropped file path
            cropped_file = os.path.join(cropped_outpur_dir, 'dem_'+ geo_file.split("/")[-1])

            # # Crop the VRT using gdalwarp
            # subprocess.run([
            #     "gdalwarp",
            #     "-te", str(min_x), str(min_y), str(max_x), str(max_y),  # Bounding box
            #     "-of", "GTiff",  # Output format
            #     merged_vrt, cropped_file
            # ])
            
            # Step 2: Crop and resample the high-resolution raster
            subprocess.run([
                "gdalwarp",
                "-te", str(min_x), str(min_y), str(max_x), str(max_y),  # Bounding box
                "-tr", str(pixel_width), str(abs(pixel_height)),       # Target resolution
                "-ts", str(cols), str(rows),                          # Target size (columns and rows)
                "-t_srs", low_res_projection,                         # Target projection
                "-r", "nearest",                                     # Resampling method (e.g., bilinear, cubic)
                "-of", "GTiff",                                       # Output format
                merged_vrt,
                cropped_file
            ])
            
            
            
            print(f"Cropped raster saved to {cropped_file}")
            
            
            
            
        #     import subprocess
        # from osgeo import gdal

        # # Input file paths
        # high_res_raster = "/path/to/high_res.tif"  # High spatial resolution raster
        # low_res_raster = "/path/to/low_res.tif"   # Low spatial resolution raster
        # output_cropped_raster = "/path/to/cropped_high_res.tif"  # Output cropped raster




        # # Step 2: Crop and resample the high-resolution raster
        # subprocess.run([
        #     "gdalwarp",
        #     "-te", str(min_x), str(min_y), str(max_x), str(max_y),  # Bounding box
        #     "-tr", str(pixel_width), str(abs(pixel_height)),       # Target resolution
        #     "-ts", str(cols), str(rows),                          # Target size (columns and rows)
        #     "-t_srs", low_res_projection,                         # Target projection
        #     "-r", "nearest",                                     # Resampling method (e.g., bilinear, cubic)
        #     "-of", "GTiff",                                       # Output format
        #     high_res_raster,
        #     output_cropped_raster
        # ])

        # print(f"Cropped high-resolution raster saved to {output_cropped_raster}")

