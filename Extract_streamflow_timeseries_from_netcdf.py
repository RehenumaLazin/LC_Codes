import xarray as xr
import numpy as np
import os

import rasterio
from rasterio.transform import from_origin
from rasterio import warp
from netCDF4 import Dataset
from osgeo import gdal
from datetime import datetime, timedelta
import pandas as pd
import glob
import calendar
import traceback
import csv


import rasterio
from rasterio.transform import from_origin,rowcol
from scipy.interpolate import griddata
from pyproj import Proj, Transformer

def get_streamflow_time_series(data_dir, output_dir, latitude, longitude, start_year=2018, end_year=2018):
    """
    Extract streamflow time series for a given latitude and longitude from daily NetCDF files.
    
    Parameters:
        data_dir (str): Directory containing daily NetCDF files.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        start_year (int): Start year of the time series.
        end_year (int): End year of the time series.
    
    Returns:
        time_series (dict): Dictionary containing time series data with timestamps.
    """
    time_series = {}
    
    for year in range(start_year, end_year + 1):
        print(year)
        os.makedirs(f"{output_dir}/{year}", exist_ok= True)
        for month in range(1, 13):
            for day in range(1, calendar.monthrange(year, month)[1] + 1):  # Handle invalid dates dynamically
                try:
                    date_str = f"{year}{month:02d}{day:02d}"
                    nc_file = f"{data_dir}{year}/{date_str}.nc"
                    if not os.path.exists(nc_file):
                        continue


                    # dataset = xr.open_dataset(nc_file)
                    nc = Dataset(nc_file, 'r')
                    flow_var = nc.variables['streamflow']  # Replace with actual variable name
                    lat = nc.variables['latitude'][:]
                    lon = nc.variables['longitude'][:]
                    
                    
                    
                    # print(flow_var[0,:].shape)
                    # time_var = nc.variables['time']

                    # Step 2: Explore the dataset and extract variables
                    # print(dataset)

                    # # Extract streamflow, latitude, and longitude
                    # streamflow = dataset['streamflow'].values  # 1D streamflow data #qSfcLatRunoff
                    # lat = dataset['latitude'].values  # 1D latitude
                    # lon = dataset['longitude'].values  # 1D longitude

                    # Step 1: Example 1D scatter data (replace with your own lat, lon, and streamflow values)
                    # lat = np.array([40.1, 40.5, 40.8, 41.0, 41.3])  # Replace with your latitudes
                    # lon = np.array([-105.3, -105.0, -104.7, -104.4, -104.1])  # Replace with your longitudes
                    # streamflow = np.array([12.5, 18.3, 15.7, 10.4, 25.1])  # Replace with your streamflow values

                    # Step 2: Define grid resolution and extent
                    grid_resolution = 0.0009  # Grid cell size (in degrees)
                    for t in range(flow_var.shape[0]):
                        
                        grid_lon, grid_lat = np.meshgrid(
                            np.arange(lon.min(), lon.max(), grid_resolution),
                            np.arange(lat.min(), lat.max(), grid_resolution)
                        )

                        # Step 3: Interpolate the scatter points onto the 2D grid
                        grid_streamflow = griddata(
                            (lon, lat),  # Points (1D arrays)
                            flow_var[t,:],  # Values at the points
                            (grid_lon, grid_lat),  # Target grid
                            method='linear'  # Linear interpolation
                        )

                        # Step 4: Define transformation and CRS (WGS84, EPSG:4326)
                        transform = from_origin(grid_lon.min(), grid_lat.max(), grid_resolution, grid_resolution)

                        # Step 5: Save the 2D grid as a GeoTIFF file using rasterio
                        
                        output_tiff = f"{output_dir}/{year}/temp_{date_str}{t:02d}.tif" #temp_raster_path = os.path.join(output_geotiff_dir, reference_raster_file.split("/")[-1].split("crop")[0][:-1] + '.tif')
                        with rasterio.open(
                            output_tiff,
                            'w',
                            driver='GTiff',
                            height=grid_streamflow.shape[0],
                            width=grid_streamflow.shape[1],
                            count=1,
                            dtype=grid_streamflow.dtype,
                            crs='EPSG:4326',  # WGS84
                            transform=transform,
                            nodata = -999900,
                        ) as dst:
                            dst.write(np.nan_to_num(np.flip(grid_streamflow, axis=0)), 1)  # Fill NaNs with zeros or appropriate values

                        print(f"\nGeoTIFF saved as: {output_tiff}")
                        

                        with rasterio.open(output_tiff) as dataset:
                            print('start reading', output_tiff)
                            # Get the coordinate reference system (CRS)
                            dataset_crs = dataset.crs

                            # Convert latitude/longitude to the dataset's coordinate system
                            transformer = Transformer.from_crs("EPSG:4326", dataset_crs, always_xy=True)
                            print(transformer)
                            x, y = transformer.transform(longitude, latitude)
                            print(x, y)

                            # Get the row and column indices of the pixel
                            row, col = rowcol(dataset.transform, x, y)
                            print(row, col)

                            # Read the pixel value
                            streamflow_value = dataset.read(1)[row, col]  # Band 1
                            print(f"Streamflow at ({latitude}, {longitude}): {streamflow_value} m³/s")
                            
                            time_series[f"{date_str}{t:02d}"] = streamflow_value
                            os.remove(output_tiff)
                            # return time_series

                            # return streamflow_value

# # Example Usage
# geotiff_file = "path_to_streamflow.tif"
# latitude = 35.5   # Example latitude
# longitude = -97.5  # Example longitude

# streamflow = get_streamflow_from_geotiff(geotiff_file, latitude, longitude)
# print(f"Streamflow at ({latitude}, {longitude}): {streamflow} m³/s")

                    
                    
                    
                #     dataset = xr.open_dataset(nc_file)
                #     lats = dataset['latitude'].values
                #     lons = dataset['longitude'].values
                #     streamflow = dataset['streamflow'].values  # Shape: (time, )
                #     times = dataset['time'].values  # Time dimension
                    
                #     lat_idx = np.abs(lats - lat).argmin()
                #     lon_idx = np.abs(lons - lon).argmin()
                    
                #     for i in range(len(times)):
                #         time_series[str(times[i])] = streamflow[i, lat_idx, lon_idx]
                    
                #     dataset.close()
                except Exception as e:
                    print(f"Skipping {date_str}: {e}")
                    print(traceback.format_exc())
                    
    with open(f"{data_dir}/sonoma_hourly.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['date', 'flow (cms)'])
        writer.writeheader()
        writer.writerow(time_series)
    

    return time_series

# Example usage
data_dir = f"/p/lustre2/lazin1/NOAA_streamflow_WRF_Hydro_HOURLY/"
output_dir = f"/p/lustre2/lazin1/NOAA_streamflow_WRF_Hydro_HOURLY/GeoTiff"
latitude = 38.4340222 # Example latitude
longitude = -123.1011083  # Example longitude
time_series = get_streamflow_time_series(data_dir, output_dir,latitude, longitude, 2017, 2019)

# Print results
for time, flow in time_series.items():
    print(f"{time}: {flow} m^3/s")
