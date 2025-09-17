import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio import warp
from netCDF4 import Dataset
from osgeo import gdal
from datetime import datetime, timedelta
import pandas as pd
import glob
import calendar
from netCDF4 import Dataset
import numpy as np


data_dir = f"/p/lustre2/lazin1/SMAP_HydroBlocks/SMAP-HydroBlocks_postprocessing/SMAPHB_sample/SMAP-HB_surface-soil-moisture_30m_daily_netcdf"
files = sorted(glob.glob(f"{data_dir}/*.nc"))
geo_dir = f"/p/lustre2/lazin1/SMAP_HydroBlocks/SMAP-HydroBlocks_postprocessing/SMAPHB_sample/SMAP-HB_surface-soil-moisture_30m_daily_geotiff"
for file in files:

    temp_raster_path = os.path.join(geo_dir, file[:-3] + '.tif')
    # Open NetCDF file
    nc = Dataset(file, 'r')
    SM_var = nc.variables['SMAPHB_SM']  # Replace with actual variable name
    latitudes = nc.variables['lat'][:]
    longitudes = nc.variables['lon'][:]
    time_var = nc.variables['time']

    # Find the start index for the 3-day period
    # time_units = time_var.units  # Expected format: "hours since YYYY-MM-DD HH:MM:SS"


    # Sum precipitation over the previous 24 hours (1 day)
    SM = SM_var[0, :, :]
    
    
    
    with rasterio.open(
        temp_raster_path,
        "w",
        driver="GTiff",
        height=SM.shape[0],
        width=SM.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(longitudes.min(), latitudes.max(), longitudes[1] - longitudes[0], latitudes[1] - latitudes[0])
    ) as temp_dst:
        temp_dst.write((np.flip(SM, axis=0)), 1)