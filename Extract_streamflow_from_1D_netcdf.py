import xarray as xr
import numpy as np

def get_streamflow(nc_file, lat, lon):
    """
    Extract streamflow value for a given latitude and longitude from a 1D NetCDF file.

    Parameters:
        nc_file (str): Path to the NetCDF file.
        lat (float): Target latitude.
        lon (float): Target longitude.

    Returns:
        float: Streamflow value at the nearest available location.
    """
    # Open the NetCDF file
    dataset = xr.open_dataset(nc_file)

    # Extract 1D lat, lon, and streamflow values
    lats = dataset['latitude'].values  # 1D latitude array
    lons = dataset['longitude'].values  # 1D longitude array
    streamflow = dataset['streamflow'].values  # 1D streamflow array

    # Find the nearest latitude and longitude indices
    lat_idx = np.abs(lats - lat).argmin()
    lon_idx = np.abs(lons - lon).argmin()

    # Ensure lat and lon indices are the same (assuming lat/lon pairs correspond to streamflow values)
    if lat_idx == lon_idx:
        streamflow_value = streamflow[lat_idx]  # Extract the corresponding streamflow value
    else:
        print("Warning: Latitude and Longitude indices do not match, using nearest neighbor approximation.")
        streamflow_value = streamflow[min(lat_idx, lon_idx)]

    # Close the dataset
    dataset.close()

    return streamflow_value

# Example Usage
nc_file = "your_file.nc"
latitude = 35.5  # Example latitude
longitude = -97.5  # Example longitude

streamflow_value = get_streamflow(nc_file, latitude, longitude)
print(f"Streamflow at ({latitude}, {longitude}): {streamflow_value} m³/s")
