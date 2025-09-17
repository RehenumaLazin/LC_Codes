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


import rasterio
from rasterio.transform import from_origin,rowcol
from scipy.interpolate import griddata
from pyproj import Proj, Transformer

def get_streamflow_from_geotiff(geotiff_file, lat, lon):
    """
    Extract streamflow value for a given latitude and longitude from a GeoTIFF file.

    Parameters:
        geotiff_file (str): Path to the GeoTIFF file.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        float: Streamflow value at the given coordinates.
    """
    with rasterio.open(geotiff_file) as dataset:
        # Get the coordinate reference system (CRS)
        dataset_crs = dataset.crs

        # Convert latitude/longitude to the dataset's coordinate system
        transformer = Transformer.from_crs("EPSG:4326", dataset_crs, always_xy=True)
        print(transformer)
        x, y = transformer.transform(lon, lat)
        print(x,y)

        # Get the row and column indices of the pixel
        row, col = rowcol(dataset.transform, x, y)
        print(row, col)

        # Read the pixel value
        streamflow_value = dataset.read(1)[row, col]  # Band 1

        return streamflow_value

# Example Usage
geotiff_file = f"/p/lustre2/lazin1/NOAA_streamflow_WRF_Hydro_HOURLY/GeoTiff/2017/temp_2017010200.tif"#f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/streamflow/flood_WM_S1A_IW_GRDH_1SDV_20180919T231415_20180919T231440_023774_0297CC_374E/flood_WM_S1A_IW_GRDH_1SDV_20180919T231415_20180919T231440_023774_0297CC_374E.tif"#f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/streamflow/non_flood_WM3_S1A_IW_GRDH_1SDV_20170910T002645_20170910T002710_018306_01ECAF_2E7C/non_flood_WM3_S1A_IW_GRDH_1SDV_20170910T002645_20170910T002710_018306_01ECAF_2E7C.tif"
latitude = 38.4340222 # Example latitude
longitude = -123.1011083  # Example longitude

streamflow = get_streamflow_from_geotiff(geotiff_file, latitude, longitude)
print(f"Streamflow at ({latitude}, {longitude}): {streamflow} m³/s")
