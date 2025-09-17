import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import os
import calendar
import traceback




def extract_streamflow(target_idxs,target_gauges,start_year, end_year):
    
    for i, gauge in enumerate(target_gauges):
        date = []
        flow_cms = []
        
        for year in range(start_year, end_year + 1):
            print(year)
            # os.makedirs(f"{output_dir}/{year}", exist_ok= True)
            for month in range(1, 13):
                for day in range(1, calendar.monthrange(year, month)[1] + 1):  # Handle invalid dates dynamically
                    try:
                        date_str = f"{year}{month:02d}{day:02d}"
                        nc_file = f"{data_dir}{year}/{date_str}.nc"
                        if not os.path.exists(nc_file):
                            continue
                        
                        # Load the NetCDF file
                        dataset = xr.open_dataset(f"/p/lustre2/lazin1/NOAA_streamflow_WRF_Hydro_HOURLY/{year}/{date_str}.nc")

                        # Extract data
                        # lats = dataset['latitude']
                        # lons = dataset['longitude']
                        streamflow = dataset['streamflow'].values
                        # lat_long_array = np.array([lats,lons]).T   # Los Angeles
                        # distances = np.sqrt((lats.values - target_lats[i])**2 + (lons.values - target_longs[i])**2)

                        # # Get index of the minimum distance (closest point)
                        # min_idx = np.argmin(distances)
                        # print(min_idx)
                        streamflow_values = streamflow[:, target_idxs[i]]
                        for t in range(streamflow_values.shape[0]):
                            date.append(f"{date_str}{t:02d}")
                            flow_cms.append(streamflow_values[t])
                            # Return the closest matching lat/lon pair
                            # tuple(lat_long_array[min_idx])

                            # # Print results
                            # for time, flow in streamflow_series.items():
                            #     print(f"{time}: {flow} m³/s")
                        # print(flow_cms)
                    except Exception as e:
                        print(f"Skipping {date_str}: {e}")
                        print(traceback.format_exc())
                        
        flow_data = pd.DataFrame({"Date": date,"Flow (cms)":flow_cms})   
        flow_data.to_csv(f"{data_dir}/Timeseries/{gauge}.csv", index=False)             
        # with open(f"{data_dir}/Timeseries/{gauge}.csv", 'w', newline='') as csvfile:
        #     # writer = csv.DictWriter(csvfile, fieldnames=['date', 'flow (cms)'])
        #     writer = csv.DictWriter(csvfile)
        #     # writer.writeheader()
        #     writer.writerow(time_series)
        # return time_series    


data_dir = f"/p/lustre2/lazin1/NOAA_streamflow_WRF_Hydro_HOURLY/"
# output_dir = f"/p/lustre2/lazin1/NOAA_streamflow_WRF_Hydro_HOURLY/Timeseries"

location = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/sonoma.csv"
df = pd.read_csv(location, delimiter=",")
target_lats = df['lat'].values
target_longs = df['long'].values
target_gauges = df['ID'].values


dataset = xr.open_dataset(f"/p/lustre2/lazin1/NOAA_streamflow_WRF_Hydro_HOURLY/2017/20170101.nc")

# Extract data
lats = dataset['latitude']
lons = dataset['longitude']
streamflow = dataset['streamflow']
target_idxs = []
for g in range(target_gauges.shape[0]):
    
# lat_long_array = np.array([lats,lons]).T   # Los Angeles
    distances = np.sqrt((lats.values - target_lats[g])**2 + (lons.values - target_longs[g])**2)

    # Get index of the minimum distance (closest point)
    target_idxs.append(np.argmin(distances))

print(target_idxs)


streamflow_series=extract_streamflow(target_idxs,target_gauges,2017, 2019)
print(streamflow_series)
           

