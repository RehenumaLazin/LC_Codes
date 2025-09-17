import os
import numpy as np
import rasterio
import pandas as pd
import pickle

import pickle


# Read GeoTIFF file as numpy array
def read_geotiff(file_path):
    with rasterio.open(file_path) as src:
        array = src.read(1)  # Read the first band
        if array.shape == (512, 512):
            return array
        else:
            print(f"Warning: {file_path} is not 512x512. Skipping...")
            return None

def save_dict(dictionary, file_path):
    """
    Saves a dictionary to a file using pickle.
    
    Args:
        dictionary (dict): The dictionary to save.
        file_path (str): Path to the file where the dictionary will be saved.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(dictionary, f)
    print(f"Dictionary saved to {file_path}")

def load_dict(file_path):
    """
    Loads a dictionary from a file using pickle.
    
    Args:
        file_path (str): Path to the file where the dictionary is saved.
    
    Returns:
        dict: The loaded dictionary.
    """
    with open(file_path, 'rb') as f:
        dictionary = pickle.load(f)
    print(f"Dictionary loaded from {file_path}")
    return dictionary


EVENT_STRS = ["Mississippi_20190617_3AC6_non_flood"] #["Florence_20180919_374E_non_flood", "Florence_20180919_B86C_non_flood"]#["Harvey_20170829_D734_non_flood","Harvey_20170831_9366_non_flood" , "Harvey_20170831_0776_non_flood", "Harvey_20170829_B8C4_non_flood", "Harvey_20170829_3220_non_flood"]
# Harvey_20170829_3220_non_flood.csv #Harvey_20170829_B8C4_non_flood.csv  #Harvey_20170831_0776_non_flood.csv #Harvey_20170831_9366_non_flood.csv #Harvey_20170829_D734_non_flood.csv
#"Mississippi_20190617_B9D7_non_flood" #"Mississippi_20190617_B9D7_non_flood"  # "Mississippi_20190617_42BF_non_flood"  #"Mississippi_20190617_3AC6_non_flood" #"Mississippi_20190617_9D85_non_flood" #"Mississippi_20190617_5E5F_non_flood"
for EVENT_STR in EVENT_STRS:
    print(EVENT_STR)
    TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_{EVENT_STR}" #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/flood_img_list_flood_non_flood_test2.csv')  #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_3AC6.csv #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_C310.csv  #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_B9D7.csv
    if os.path.exists(f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_input_dict_prec_one_fourth_not_30.pkl") and os.path.exists(f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_target_dict.pkl"): #CHANGE for PREC DOUBLED _prec_doubled
        print(f"{EVENT_STR} exists, moving to next")
        continue
    else:
        

    # Define the path to the folder containing your target GeoTIFF files
    # EVENT_STR = "Mississippi_20190617_C310_non_flood" #"Mississippi_20190617_C310_non_flood"  #"Mississippi_20190617_B9D7_non_flood"  # "Mississippi_20190617_42BF_non_flood"  #"Mississippi_20190617_3AC6_non_flood" #"Mississippi_20190617_9D85_non_flood" #"Mississippi_20190617_5E5F_non_flood"
    # TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_{EVENT_STR}" #'/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood'  # Reference raster for extent and resolution

        INPUT_VARIABLE_DIRS = [f"/p/lustre2/lazin1/cropped_dems_{EVENT_STR}" , #Mississippi_20190617_5E5F_cropped_dems_non_flood
                        f"/p/lustre2/lazin1/cropped_LC_{EVENT_STR}",   #/p/lustre2/lazin1/cropped_LC_Mississippi_20190617_5E5F_non_flood',
                        f"/p/lustre2/lazin1/AORC_APCP_surface/1_day_AORC_Prec_{EVENT_STR}",  #1_day_AORC_Prec_Tile_Mississippi_20190617_5E5F_non_flood',
                        f"/p/lustre2/lazin1/AORC_APCP_surface/5_day_AORC_Prec_{EVENT_STR}",  #5_day_AORC_Prec_Mississippi_20190617_5E5F_non_flood/',
                        f"/p/lustre2/lazin1/AORC_APCP_surface/30_day_AORC_Prec_{EVENT_STR}",  #30_day_AORC_Prec_Mississippi_20190617_5E5F_non_flood'
                        ]


        # Define the path to the csv containing your target GeoTIFF files
        TILE_LIST_CSV = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/{EVENT_STR}.csv"   #"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Mississippi_20190617_5E5F_non_flood.csv"




        # Define dir to save means and stds
        mean_std_dir = f"/p/vast1/lazin1/UNet_inputs/mean_stds"



        # Read tile GeoTIFF files as numpy array
        tile_files = pd.read_csv(TILE_LIST_CSV).to_numpy()



        # Initialize the array to store all data (assuming 4223 samples, 5 variables, 512x512 each)
        num_samples =  tile_files.shape[0]  #9119#1823   # 9119 #1880
        num_variables = 5
        input_array_shape = (num_samples, num_variables, 512, 512)
        input_array = np.empty(input_array_shape, dtype=np.float32)

        target_array_shape = (num_samples, 1, 512, 512)
        target_array = np.empty(target_array_shape)

        # Define the variable types and ensure they are identifiable in filenames
        variable_types = ['DEM', 'LC', 'Prec1', 'Prec5', 'Prec30']


        var_strs = ['dem_', #dem_
                'LC_',
                '1day_prec_',
                '5day_prec_',
                '30day_prec_'
            
        ]
            


                

        for i, tile_file in enumerate(tile_files):
            # print(tile_file[1])
            for j, variable_type in enumerate(variable_types):
                var_array = read_geotiff(os.path.join(INPUT_VARIABLE_DIRS[j],var_strs[j] + tile_file[1]))
                input_array[i,j] = var_array
                
            target_array[i,0] = read_geotiff(os.path.join(TARGET_RASTER_DIR , tile_file[1]))
            print(i)



        # save mean and stds of the required varsiables
        dem_mean = np.mean(input_array[:,0,:,:], axis=0)
        np.save(f"{mean_std_dir}/{EVENT_STR}_dem_mean.npy", dem_mean)

        dem_std = np.std(input_array[:,0,:,:], axis=0)
        np.save(f"{mean_std_dir}/{EVENT_STR}_dem_std.npy", dem_std)


        d1_prec_mean = np.mean(input_array[:,2,:,:], axis=0)
        np.save(f"{mean_std_dir}/{EVENT_STR}_1d_prec_mean.npy", d1_prec_mean)

        d1_prec_std = np.std(input_array[:,2,:,:], axis=0)
        np.save(f"{mean_std_dir}/{EVENT_STR}_1d_prec_std.npy", d1_prec_std)

        d5_prec_mean = np.mean(input_array[:,3,:,:], axis=0)
        np.save(f"{mean_std_dir}/{EVENT_STR}_5d_prec_mean.npy", d5_prec_mean)

        d5_prec_std = np.std(input_array[:,3,:,:], axis=0)
        np.save(f"{mean_std_dir}/{EVENT_STR}_5d_prec_std.npy", d5_prec_std)

        d30_prec_mean = np.mean(input_array[:,4,:,:], axis=0)
        np.save(f"{mean_std_dir}/{EVENT_STR}_30d_prec_mean.npy", d30_prec_mean)

        d30_prec_std = np.std(input_array[:,4,:,:], axis=0)
        np.save(f"{mean_std_dir}/{EVENT_STR}_30d_prec_std.npy", d30_prec_std)

        input_array_normalized = np.empty(input_array_shape, dtype=np.float32)
        input_array_normalized[:,0,:,:] = (input_array[:,0,:,:] - dem_mean) / dem_std
        input_array_normalized[:,1,:,:] =  np.where(input_array[:,1,:,:] == 11, 1, 0)     
        # input_array_normalized[:,2,:,:] = (input_array[:,2,:,:] - d1_prec_mean) / d1_prec_std
        # input_array_normalized[:,3,:,:] = (input_array[:,3,:,:] - d5_prec_mean) / d5_prec_std
        input_array_normalized[:,4,:,:] = (input_array[:,4,:,:] - d30_prec_mean) / d30_prec_std
        
        input_array_normalized[:,2,:,:] = (input_array[:,2,:,:]*0.25 - d1_prec_mean) / d1_prec_std #CHANGE for PREC DOUBLED
        input_array_normalized[:,3,:,:] = (input_array[:,3,:,:]*0.25 - d5_prec_mean) / d5_prec_std #CHANGE for PREC DOUBLED
        # input_array_normalized[:,4,:,:] = (input_array[:,4,:,:]*1.5 - d30_prec_mean) / d30_prec_std #CHANGE for PREC DOUBLED

        input_data = {}  
        target_data = {}
        for i, tile_file in enumerate(tile_files):
            input_data[tile_file[1]] = input_array_normalized[i]
            
            target_data[tile_file[1]] = target_array[i]
        
        # np.save('/p/vast1/lazin1/UNet_inputs/input_data_normalized_20190617_5E5F_non_flood_5_var.npy', data_array_normalized) 
        save_dict(input_data, f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_input_dict_prec_one_fourth_not_30.pkl") #CHANGE for PREC DOUBLED _prec_doubled
        save_dict(target_data, f"/p/vast1/lazin1/UNet_inputs/{EVENT_STR}_target_dict.pkl")
