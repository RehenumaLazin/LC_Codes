import pandas as pd
import glob
import sys

def combine_csv_files(input_folder, output_file):
    """
    Reads all CSV files from the specified folder and combines them into one CSV file.
    
    Args:
        input_folder (str): The folder containing the CSV files.
        output_file (str): The path to the output combined CSV file.
    """
    # Get all CSV file paths in the folder
    print(input_folder)
    csv_files = glob.glob(f"{input_folder}/*.csv") #glob.glob(f"{TARGET_RASTER_DIR}/*.tif") 
    
    # Check if CSV files are found
    if not csv_files:
        print(f"No CSV files found in the folder: {input_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV files. Combining them...")
    
    # Read and concatenate all CSV files
    combined_df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)
    # events = [i.split("/")[-1][:-4] for i in combined_df[0]]
    # print(events)
    # events = [combined_df[i][1].split("/")[-1][:-4] for i, _ in enumerate(combined_df)]

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")
    
def combine_csv_files_from_lists_of_csv(events_file, output_file):
    
    
    combined_df = pd.read_csv(events_file, header=None) 
    csv_files=[]
    for raster_path in combined_df[0]: #events = [raster_path.split("/")[-1][:-4] 
        event = raster_path.split("/")[-1][:-4]

        csv_files.append(f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_waterbody/shapefiles/{event}.csv")  # output shapefile path #/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_waterbody/shapefiles  #/p/lustre2/lazin1/shapefiles
    combined_csvs= pd.concat((pd.read_csv(csv_file) for csv_file in csv_files), ignore_index=True)
    # events = [i.split("/")[-1][:-4] for i in combined_df[0]]
    # print(events)
    # events = [combined_df[i][1].split("/")[-1][:-4] for i, _ in enumerate(combined_df)]

    # Save the combined DataFrame to a new CSV file
    combined_csvs.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")

# Example usage

if sys.argv[1] == "folder":
    input_folder = "/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_waterbody/shapefiles"  #"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS"  # Replace with your actual folder path
    output_file = "/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_waterbody/shapefiles/combined.csv"   #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv"  # Replace with your desired output file path

    combine_csv_files(input_folder, output_file)
elif sys.argv[1] == "list":
    events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/flood_events.csv" #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined_test.csv' #combined_reduced_events_files.csv
    output_file = "/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined_flood_events_all.csv"   #f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv"  # Replace with your desired output file path
    combine_csv_files_from_lists_of_csv(events_file, output_file)
elif sys.argv[1] == "multiple_list":
    EVENT_STRS = ["Mississippi_20190617_5E5F_non_flood", 
                  "Mississippi_20190617_42BF_non_flood" ,
                  "Mississippi_20190617_9D85_non_flood", 
                  "Mississippi_20190617_3AC6_non_flood", 
                  "Harvey_20170829_D734_non_flood",
                  "Harvey_20170831_9366_non_flood" , 
                  "Harvey_20170831_0776_non_flood", 
                  "Harvey_20170829_B8C4_non_flood", 
                  "Harvey_20170829_3220_non_flood", 
                  "Florence_20180919_B86C_non_flood", 
                  "Mississippi_20190617_C310_non_flood",
                  "Mississippi_20190617_B9D7_non_flood" , 
                  "Florence_20180919_374E_non_flood"]
    for EVENT_STR in EVENT_STRS:
        events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/EVENT_{EVENT_STR}.csv" 
        output_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/combined_{EVENT_STR}.csv" 


        combine_csv_files_from_lists_of_csv(events_file, output_file)


