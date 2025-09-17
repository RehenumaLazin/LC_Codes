import rasterio
import pandas as pd
import glob
import os
from datetime import datetime, timedelta

# Paths
gauge_file = "/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Guadalupe_River/Guadalupe_USGS_gauge_height_in_meter/guadalupe_river.csv"#"/usr/WS1/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_gauge_height.csv"
event_file = "/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Guadalupe_River/guadalupe_river.csv" #"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_events.csv"

# Read CSV files
gauge_df = pd.read_csv(gauge_file)
event_df = pd.read_csv(event_file)

# Extract columns
lats = gauge_df['Lat'].values
lons = gauge_df['Lon'].values
stationIDs = gauge_df['STAID'].values
start_dates = event_df['start']
end_dates = event_df['end']
print(start_dates)

# Loop over events
for event_index in range(len(start_dates)):
    # if event_index in range(9):
    #     continue
    
    start_str = start_dates[event_index]
    print(start_str)
    end_str = end_dates[event_index]
    tif_dir = f"/p/lustre1/lazin1/flood/Texas/GuadalupeRiver_2025-07-02_2025-07-06/"#f"/p/lustre1/lazin1/flood/Sonoma/RussianRiver_{start_str}_{end_str}/"
    
    files = sorted(glob.glob(os.path.join(tif_dir, "depth-001-*.tif")), key=os.path.getmtime)
    print(len(files), (os.path.join(tif_dir, "depth-1-*.tif")))
    
    start_time = datetime.strptime(start_str, '%Y-%m-%d')

    # Initialize data storage for each station
    station_data = {sta: [] for sta in stationIDs}

    for file in files:
        try:
            minutes = int(os.path.basename(file).split("-")[-1].split(".")[0])
            print(minutes)
        except ValueError:
            continue

        timestamp = start_time + timedelta(minutes=minutes)
        # date = datetime.fromtimestamp(timestamp).date().strftime('%Y%m%d%H'))

        with rasterio.open(file) as src:
            for i, sta in enumerate(stationIDs):
                try:
                    row, col = src.index(lons[i], lats[i])
                    value = src.read(1)[row, col]
                except IndexError:
                    value = None

                station_data[sta].append({"date": timestamp, "Torrent_flood_depth(m)": value})

    # Save CSV per station for this event
    for sta in stationIDs:
        df = pd.DataFrame(station_data[sta])
        output_filename = f"/p/lustre1/lazin1/flood/Texas/GuadalupeRiver_2025-07-02_2025-07-06/Torrent_with_date/{sta}_TORRENT.csv"#f"/p/lustre1/lazin1/flood/Sonoma/Torrent_with_date/event{event_index+1}/{sta}_TORRENT.csv"
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        df.to_csv(output_filename, index=False)
        print(f"Saved: {output_filename}")
        
