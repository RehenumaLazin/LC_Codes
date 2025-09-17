import rasterio
import pandas as pd
import glob
import os
from datetime import datetime, timedelta

# Paths
gauge_file = "/usr/WS1/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_gauge_height.csv"
event_file = "/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_events.csv"

# Read CSV files
gauge_df = pd.read_csv(gauge_file)
event_df = pd.read_csv(event_file)

# Extract columns
lats = gauge_df['Lat'].values
lons = gauge_df['Long'].values
stationIDs = gauge_df['STAID'].values
start_dates = event_df['start']
end_dates = event_df['end']

# Loop over events
for event_index in range(len(start_dates)):
    if event_index not in [1,]: continue
    # start_str = start_dates[event_index]
    # end_str = end_dates[event_index] #/p/vast1/lazin1/UNet_Geotiff_output/Sonoma_event2_output_not_normalized_retrain
    tif_dir = f"/p/vast1/lazin1/UNet_Geotiff_output/Sonoma_event2_prec_dem_LC_Hourly_script_fold11/test/" #f"/p/lustre1/lazin1/flood/Sonoma/RussianRiver_{start_str}_{end_str}/" /p/vast1/lazin1/UNet_Geotiff_output/Sonoma_event2_prec_dem_LC_Hourly_script_LSTM_UNet_fold11
    
    # files = sorted(glob.glob(os.path.join(tif_dir, "depth-1-*.tif")), key=os.path.getmtime)
    # Pattern to match daily files
    files = sorted(glob.glob(os.path.join(tif_dir, f"merged_2019*_Sonoma.tif")))
    print(len(files), files)
    
    # start_time = datetime.strptime(start_str, '%Y%m%d%H')

    # Initialize data storage for each station
    station_data = {sta: [] for sta in stationIDs}

    for file in files:
        # print(file)
        date_str = file.split("/")[-1].split("_")[1] #date_str = file.split("_")[-3]
        # print(date_str)
        # print(datetime.strptime(date_str, "%Y%m%d%H").strftime("%Y-%m-%d %H:%M:%S"))
        try:
            timestamp = datetime.strptime(date_str, "%Y%m%d%H").strftime("%Y-%m-%d %H:%M:%S")
            # print(timestamp)
            
        except ValueError:
            continue  # Skip files that don't match the date format

        # timestamp = start_time + timedelta(minutes=24*60*(event_index+1))

        with rasterio.open(file) as src:
            for i, sta in enumerate(stationIDs):
                try:
                    row, col = src.index(lons[i], lats[i])
                    value = src.read(1)[row, col]
                except IndexError:
                    value = None

                station_data[sta].append({"date": timestamp, "UNet_flood_depth(m)": value})

    # Save CSV per station for this event
    for sta in stationIDs:
        df = pd.DataFrame(station_data[sta])
        output_filename = f"/p/vast1/lazin1/UNet_Geotiff_output/csv/event{event_index+1}/{sta}_UNet_hourly_script_LSTM_UNet_fold11_0909.csv"
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        df.to_csv(output_filename, index=False)
        print(f"Saved: {output_filename}")








