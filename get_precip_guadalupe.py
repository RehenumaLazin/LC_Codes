import numpy as np
import h5py
import xcdat as xc
import xarray as xa
import cftime
import glob
import rasterio
from rasterio.transform import from_origin
from datetime import datetime, timedelta
import json
import os
import subprocess

import pandas as pd

import h5py
import numpy as np
import xarray as xa
import rasterio
from rasterio.transform import from_origin
from datetime import datetime
import os
import glob

events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Guadalupe_River/guadalupe_river.csv" #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv' #combined.csv'
combined_df = pd.read_csv(events_file) 
start_dates = combined_df['start'].to_numpy()
end_dates = combined_df['end'].to_numpy() 

path = '/p/lustre2/lazin1/IMERG/'

year_months = [y_m for y_m in os.listdir(path) if os.path.isdir(os.path.join(path, y_m))]

# Function to open and read HDF5 files
def open_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        # Access the dataset (assuming the dataset is within 'Grid')
        grid_data = f['Grid/precipitation'][:]  # Replace with correct dataset path
        time_data = f['Grid']['time'] [:] # Replace with actual time variable path
        print(time_data)
        ref = f['Grid']['time'].attrs['units']
        print(ref)

        reference_time = datetime.strptime(str(ref).split('since ')[1].replace(' UTC','')[:-1], '%Y-%m-%d %H:%M:%S')
        print(reference_time)
        time_data_converted = [reference_time + timedelta(seconds=int(t)) for t in time_data]

        # time_data = [np.datetime64(t, 'ns').astype(datetime) for t in time_data]
        lat_data = f['Grid']['lat'][:] # Replace with actual latitude variable path
        lon_data = f['Grid']['lon'][:]  # Replace with actual longitude variable path
        
        # Convert time data to datetime
        # time_data = [datetime.utcfromtimestamp(t) for t in time_data]  # Adjust if necessary


        print(f"Converted Time Data (first 5 entries): {time_data[:5]}")
        
    return grid_data, lat_data, lon_data, time_data_converted


 

for idx, (s,e) in enumerate(zip(start_dates, end_dates)):
    print(idx)
    
    start_dt = datetime.strptime(s +" 00", "%Y-%m-%d %H")
    start_date = start_dt.strftime( "%Y-%m-%d %H")
    end_dt = datetime.strptime(e +" 23", "%Y-%m-%d %H")
    end_date = end_dt.strftime( "%Y-%m-%d %H")

    start_ym = start_date[:4]+'_'+start_date[5:7] 
    end_ym = end_date[:4]+'_'+end_date[5:7]
    print(start_ym, end_ym)
    print(start_date, end_date)
    file_pattern = "3B-HHR-L.MS.MRG.3IMERG*.HDF5"  # Modify if files are in a different directory
    # all_files = sorted(glob.glob(path+start_ym+file_pattern))  # Get all matching files
    # print(len(all_files))
    # # Filter files based on year-month
    # filtered_files = []
    # for file in all_files:
    #     year = file.split("3IMERG.")[-1][:4]
    #     month = file.split("3IMERG.")[-1][4:6]  # Extract YYYY_MM
    #     year_month = year + "_" + month
    #     if start_ym <= year_month <= end_ym:
    #         filtered_files.append(file)
    # print(filtered_files)
    filtered_files = []
    for y_m in year_months:
        if start_ym <= y_m <= end_ym:
            filtered_files.extend(sorted(glob.glob(path+start_ym+'/'+file_pattern)))
            
    print(len(filtered_files))
        

    # List of HDF5 files (replace with your file paths)
    hdf5_files = filtered_files #['file1.HDF5', 'file2.HDF5']  # Add the paths of the HDF5 files

    # Filters for latitude and longitude (change these ranges based on your region of interest)
    
    lat_min, lat_max = 29.75, 30.37
    lon_min, lon_max = -99.8, -98.34
    # lat_min, lat_max = 38.26, 39.5 #29.8, 30.27
    # lon_min, lon_max = -123.5, -122.4 #-99.7, -98.37
# (38.26, 39.5), longitude=slice(-123.5, -122.4)


    precip = []
    time_list = []
    lat_list = []
    lon_list = []

    for file in hdf5_files:
        print(f"Reading {file}...")
        
        # Open the HDF5 file and extract data
        grid_data, lat_data, lon_data, time_data = open_hdf5_file(file)
        
        # Slice the data based on latitude and longitude
        lat_mask = (lat_data >= lat_min) & (lat_data <= lat_max)
        lon_mask = (lon_data >= lon_min) & (lon_data <= lon_max)
        print("grid_data",grid_data.shape)
        print("lat_mask",lat_mask.shape)
        print("lon_mask",lon_mask.shape)
        # Apply the lat_mask to the latitude axis and lon_mask to the longitude axis
        grid_data_subset = grid_data[:,lon_mask, :]  # Apply latitude mask along rows
        grid_data_subset = grid_data_subset[:,:, lat_mask]  # Apply longitude mask along columns
        print(grid_data_subset.shape)

        lat_data_masked = lat_data[lat_mask]
        lon_data_masked = lon_data[lon_mask]
        print(lat_data_masked)
        print(lon_data_masked)
        grid_data_transposed = np.transpose(grid_data_subset, (0, 2, 1))
        
        # Create xarray Dataset for each time slice (optional)
        ds = xa.DataArray(grid_data_transposed, dims=["time",  "latitude", "longitude",], coords={"time": time_data, "latitude": lat_data_masked, "longitude": lon_data_masked})
        
        precip.append(ds)
        
        
    # Open the HDF5 file and extract data
    # grid_data, lat_data, lon_data, time_data = open_hdf5_file(file)

    # Check if latitude values are decreasing
    lat_diff = np.diff(lat_data_masked)
    if np.all(lat_diff >= 0):
        print("Latitudes are increasing.")
    elif np.all(lat_diff <= 0):
        print("Latitudes are decreasing.")
    else:
        print("Latitudes are neither strictly increasing nor strictly decreasing.")


    # Concatenate along the time dimension
    precip = xa.concat(precip, dim="time")
    _, index = np.unique(precip["time"], return_index=True)
    precip = precip.isel(time=index)
    print(precip)
    print(precip["time"])
    precip = precip.sel(time=slice(start_date, end_date))

    # Print the final concatenated dataset
    print(precip)

    # Now, save it as a multi-band GeoTIFF
    data_var = precip.fillna(0)  # Fill NaN values with 0
    if np.all(lat_diff >= 0):
        prec_combined = data_var.isel(latitude=slice(None, None, -1))  # Flip latitude
    else:
        prec_combined = data_var
    # prec_combined = data_var

    # Get the bounding coordinates for the map
    lat_min, lat_max = precip.latitude.min().values, precip.latitude.max().values
    lon_min, lon_max = precip.longitude.min().values, precip.longitude.max().values

    # Calculate the pixel size for the transform
    xsize = (lon_max - lon_min) / len(precip.longitude)
    ysize = (lat_max - lat_min) / len(precip.latitude)

    # Create the transform for the GeoTIFF
    transform = from_origin(west=lon_min, north=lat_max, xsize=xsize, ysize=ysize)

    # Save as a multi-band GeoTIFF
    output_file = path + "precip_IMERG_GuadalupeRiver.tif"
    with rasterio.open(
        output_file, 'w',
        driver='GTiff',
        height=prec_combined.shape[1],
        width=prec_combined.shape[2],
        count=prec_combined.shape[0],
        dtype=prec_combined.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        for i in range(prec_combined.shape[0]):
            dst.write(prec_combined.isel(time=i).values, i + 1)  # Write each time slice as a band

    print(f"GeoTIFF saved as: {output_file}")

