import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio import warp
import json
import glob
import os
import pandas as pd
import sys

def write_geotiff_from_array(output_path, array, reference_tiff_path):
    """
    Writes a numpy array as a GeoTIFF, using the spatial properties of a reference GeoTIFF.

    Parameters:
    - output_path (str): Path to save the output GeoTIFF.
    - array (numpy array): Numpy array to be written to the GeoTIFF.
    - reference_tiff_path (str): Path to the reference GeoTIFF to copy metadata from.
    """
    # Open the reference GeoTIFF to retrieve spatial metadata
    with rasterio.open(reference_tiff_path) as ref:
        # Get metadata from reference GeoTIFF
        meta = ref.meta.copy()

    # Update metadata to match the numpy array dimensions
    meta.update({
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,  # Assuming a single-band array; update if needed
        "dtype": array.dtype
    })

    # Write the array to a new GeoTIFF
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(array, 1)  # Write to the first band
        
        
        

INPUT_NPY_DIR = "/p/vast1/lazin1/UNet_npy_output"
if sys.argv[1] == "test":
    OUTPUT_GEOTIFF_DIR = "/p/vast1/lazin1/UNet_Geotiff_output"
    model_type = ""
elif sys.argv[1] == "test_event_wise":
    model_type = "_event_wise"
    OUTPUT_GEOTIFF_DIR = "/p/vast1/lazin1/UNet_Geotiff_output_event_wise"
    
elif sys.argv[1] == "test_event_wise_prec_one_fourth_not_30": 
    model_type = "_event_wise_prec_one_fourth_not_30"
    OUTPUT_GEOTIFF_DIR = "/p/vast1/lazin1/UNet_Geotiff_output_event_wise_prec_one_fourth_not_30"




EVENT_STRS = ["Mississippi_20190617_5E5F_non_flood", "Mississippi_20190617_9D85_non_flood", "Mississippi_20190617_3AC6_non_flood", "Harvey_20170829_D734_non_flood","Harvey_20170831_9366_non_flood" , "Harvey_20170831_0776_non_flood", "Harvey_20170829_B8C4_non_flood", "Harvey_20170829_3220_non_flood","Florence_20180919_B86C_non_flood", "Mississippi_20190617_C310_non_flood"] #"Mississippi_20190617_B9D7_non_flood" #"Florence_20180919_374E_non_flood", 
# EVENT_STRS = ["Florence_20180919_374E_non_flood", "Florence_20180919_B86C_non_flood"]
EVENT_STRS = ["Mississippi_20190617_3AC6_non_flood"]

for e, EVENT_STR in enumerate(EVENT_STRS):
    os.makedirs(f"{OUTPUT_GEOTIFF_DIR}/{EVENT_STR}", exist_ok=True)
    print(EVENT_STR)
    reference_raster_dir = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_{EVENT_STR}" #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/flood_img_list_flood_non_flood_test2.csv')  #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_3AC6.csv #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_C310.csv  #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_B9D7.csv

    event_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_{EVENT_STR}.csv" 
    events = pd.read_csv(event_file, header=None).to_numpy()



# EVENT_STR = "Mississippi_20190617_5E5F_non_flood"
# os.makedirs(f"{OUTPUT_GEOTIFF_DIR}/{EVENT_STR}", exist_ok=True)
            
# EVENT_FILE = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/flood_img_list_flood_non_flood_test.csv"
# events = pd.read_csv(EVENT_FILE, header=None).to_numpy()
# reference_raster_dir = '/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood'  # Reference raster for extent and resolution




    for event in events:
        event_str= event[0].split("/")[-1][:-4]
        os.makedirs(f"{OUTPUT_GEOTIFF_DIR}/{EVENT_STR}/{event_str}", exist_ok=True)
        reference_raster_files = pd.read_csv(f"{INPUT_NPY_DIR}/{EVENT_STR}/{event_str}.csv", header=None) #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Harvey_D734_Flood_imgs_tiles.csv
        reference_raster_files= reference_raster_files.to_numpy()
    # f"{INPUT_NPY_DIR}/{EVENT_STR}/{event_str}.npy"
        loaded_arrays = np.load(f"{INPUT_NPY_DIR}/{EVENT_STR}/{event_str}{model_type}.npz")  #test_result_Florence_B86C_fold_1.npy #
        print(loaded_arrays)
        array = loaded_arrays['arr_0']
        reshaped_array = np.squeeze(array, axis=1)

        for i in range(len(reference_raster_files)):
            reference_tiff_path = os.path.join(reference_raster_dir,reference_raster_files[i][0])
            
            output_GeoTIFF = os.path.join(OUTPUT_GEOTIFF_DIR,EVENT_STR, event_str, reference_raster_files[i][0])
            write_geotiff_from_array(output_GeoTIFF, reshaped_array[i, :,:], reference_tiff_path)
    
    #     # reference_tiff_path = reference_raster_file#.split("/")[-1] #reference_raster_file.split("/")[-1]
    #     write_geotiff_from_array(output_GeoTIFF, reshaped_array[i, :,:], reference_raster_file)