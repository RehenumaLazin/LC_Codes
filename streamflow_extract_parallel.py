import xarray as xr
import numpy as np
import os
import rasterio
from rasterio.transform import from_origin, rowcol
from scipy.interpolate import griddata
from pyproj import Transformer
from datetime import datetime, timedelta
import pandas as pd
import glob
import calendar
import traceback
import csv
from multiprocessing import Pool, cpu_count

def process_day(args):
    data_dir, output_dir, latitude, longitude, year, month, day = args
    time_series = {}
    try:
        date_str = f"{year}{month:02d}{day:02d}"
        nc_file = f"{data_dir}{year}/{date_str}.nc"
        if not os.path.exists(nc_file):
            return {}
        
        nc = xr.open_dataset(nc_file)
        flow_var = nc['streamflow'].values
        lat = nc['latitude'].values
        lon = nc['longitude'].values
        
        grid_resolution = 0.0009
        for t in range(flow_var.shape[0]):
            grid_lon, grid_lat = np.meshgrid(
                np.arange(lon.min(), lon.max(), grid_resolution),
                np.arange(lat.min(), lat.max(), grid_resolution)
            )
            
            grid_streamflow = griddata(
                (lon, lat), flow_var[t, :],
                (grid_lon, grid_lat), method='linear'
            )
            
            transform = from_origin(grid_lon.min(), grid_lat.max(), grid_resolution, grid_resolution)
            output_tiff = f"{output_dir}/{year}/temp_{date_str}{t:02d}.tif"
            
            with rasterio.open(
                output_tiff, 'w', driver='GTiff',
                height=grid_streamflow.shape[0], width=grid_streamflow.shape[1],
                count=1, dtype=grid_streamflow.dtype,
                crs='EPSG:4326', transform=transform, nodata=-999900
            ) as dst:
                dst.write(np.nan_to_num(np.flip(grid_streamflow, axis=0)), 1)
            
            with rasterio.open(output_tiff) as dataset:
                transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
                x, y = transformer.transform(longitude, latitude)
                row, col = rowcol(dataset.transform, x, y)
                streamflow_value = dataset.read(1)[row, col]
                time_series[f"{date_str}{t:02d}"] = streamflow_value
                os.remove(output_tiff)
    except Exception as e:
        print(f"Skipping {date_str}: {e}")
        print(traceback.format_exc())
    return time_series

def get_streamflow_time_series(data_dir, output_dir, latitude, longitude, start_year=2018, end_year=2018):
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = []
    for year in range(start_year, end_year + 1):
        os.makedirs(f"{output_dir}/{year}", exist_ok=True)
        for month in range(1, 13):
            for day in range(1, calendar.monthrange(year, month)[1] + 1):
                tasks.append((data_dir, output_dir, latitude, longitude, year, month, day))
    
    time_series = {}
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_day, tasks)
        for result in results:
            time_series.update(result)
    
    with open(f"{data_dir}/sonoma_hourly.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['date', 'flow (cms)'])
        writer.writeheader()
        for key, value in time_series.items():
            writer.writerow({'date': key, 'flow (cms)': value})
    
    return time_series

# Example usage
data_dir = f"/p/lustre2/lazin1/NOAA_streamflow_WRF_Hydro_HOURLY/"
output_dir = f"/p/lustre2/lazin1/NOAA_streamflow_WRF_Hydro_HOURLY/GeoTiff"
latitude = 38.4340222 
longitude = -123.1011083  
time_series = get_streamflow_time_series(data_dir, output_dir, latitude, longitude, 2017, 2019)

for time, flow in time_series.items():
    print(f"{time}: {flow} m^3/s")
