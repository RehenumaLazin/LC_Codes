import xarray as xr
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import pandas as pd

# Function to get bounding box from GeoTIFF file
def get_geotiff_bounds(geotiff_file):
    with rasterio.open(geotiff_file) as src:
        bounds = src.bounds
        print(bounds)
        return bounds

# Function to crop the NetCDF file based on bounding box
def crop_netcdf(netcdf_file, bounds):
    # Load the NetCDF file
    ds = xr.open_dataset(netcdf_file)
    
    # Convert bounding box to lat/lon ranges
    lon_min, lat_min, lon_max, lat_max = bounds.left, bounds.bottom, bounds.right, bounds.top
    
    # Check if latitudes are reversed
    lat_dim = ds['latitude'].values
    if lat_dim[0] > lat_dim[-1]:  # Latitudes are decreasing
        cropped_ds = ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
    else:
        cropped_ds = ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
    
    return cropped_ds

# Function to calculate regional average precipitation and plot
def average_and_plot(cropped_ds):
    # Calculate the regional average (averaging over lat and lon dimensions)
    regional_avg = cropped_ds['APCP_surface'].mean(dim=['latitude', 'longitude'])
    
    # Plotting the time series
    plt.figure(figsize=(10, 6))
    regional_avg.plot()
    plt.title('Regional Average Hourly Precipitation')
    plt.xlabel('Time')
    plt.ylabel('Precipitation (mm/hr)')
    plt.grid(True)
    plt.savefig("/p/vast1/lazin1/UNet_inputs/daily_accum_prec/plt.png")
    plt.show()

# Main workflow
daily_prec=[]
events_file = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined_test.csv'
combined_df = pd.read_csv(events_file, header=None) 
for idx, raster_path in enumerate(combined_df[0]): #events = [raster_path.split("/")[-1][:-4] 
    event = raster_path.split("/")[-1][:-4]
def main():
    netcdf_file = os.path.join('/p/lustre2/lazin1/AORC_APCP_surface/APCP_surface_'+ '2019' + '_' + '06'+ '.nc') 
    geotiff_file = raster_path #f"/p/vast1/lazin1/UNet_inputs/daily_accum_prec/{event}"
    
    # Get bounding box from GeoTIFF
    bounds = get_geotiff_bounds(geotiff_file)
    
    # Crop the NetCDF file
    cropped_ds = crop_netcdf(netcdf_file, bounds)
    
    # Calculate regional average and plot the time series
    average_and_plot(cropped_ds)

if __name__ == "__main__":
    main()