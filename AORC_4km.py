import xarray as xr
import rioxarray
from rioxarray.rio import reproject_match
from rioxarray.rio.enums import Resampling

# -------------------------------
# Step 1: Load the 10 m precipitation data (multiple monthly files)
# -------------------------------
precip_files = "/p/lustre2/lazin1/AORC_APCP_surface/APCP_surface_*.nc"  # Glob pattern for all monthly files
precip_ds = xr.open_mfdataset(precip_files, combine='by_coords')

# Set spatial dimensions and CRS (adjust x/y or lon/lat as needed)
precip_ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
precip_ds.rio.write_crs("EPSG:4326", inplace=True)  # Replace with actual CRS if different

# -------------------------------
# Step 2: Load the reference 4 km resolution dataset
# -------------------------------
reference_ds = xr.open_dataset("reference_4km.nc")

# Set CRS and spatial dims if needed
reference_ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
reference_ds.rio.write_crs("EPSG:4326", inplace=True)

# -------------------------------
# Step 3: Reproject and aggregate
# -------------------------------
# Replace 'precip' with your actual variable name
precip_var = precip_ds['precip']  # Change 'precip' to the correct variable name if needed

# Match resolution and extent, use sum for precipitation accumulation
resampled_precip = precip_var.rio.reproject_match(reference_ds, resampling=Resampling.sum)

# -------------------------------
# Step 4: Save output to NetCDF
# -------------------------------
resampled_precip.to_netcdf("precip_hourly_aggregated_4km.nc")

print("Aggregation complete. File saved as 'precip_hourly_aggregated_4km.nc'")
