import subprocess
from osgeo import ogr
import os

# ----- User-defined file paths -----
dem_vrt = "/p/lustre1/lazin1/USGS_DEM_10m/merged_USGS_DEM_10m.vrt"                     # Input DEM VRT file
shapefile = "/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Guadalupe_River/Guadalupe_River.shp"#"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/11467270.shp"              # Shapefile defining the crop extent
output_geotiff = "/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Guadalupe_River/Guadalupe_dem_near_kerr.tif"         # Output cropped GeoTIFF
cropped_outpur_dir = f"/p/lustre2/lazin1/DEM"
# output_geotiff = os.path.join(cropped_outpur_dir, f"cropped_dem_11467270_extent.tif")
output_geotiff = os.path.join(cropped_outpur_dir, f"Guadalupe_dem_near_kerr.tif")
# ---------------------------------------

# Step 1: Open the shapefile and extract its bounding box.
# OGR's GetExtent() returns (minX, maxX, minY, maxY)
ds = ogr.Open(shapefile)
layer = ds.GetLayer()
extent = layer.GetExtent()
ds = None  # Close the shapefile

# gdalwarp expects: minX, minY, maxX, maxY
minX = extent[0]
maxX = extent[1]
minY = extent[2]
maxY = extent[3]

print("Extracted bounding box from shapefile:")
print("minX =", minX, "minY =", minY, "maxX =", maxX, "maxY =", maxY)

# Step 2: Use gdalwarp to crop the DEM VRT to the bounding box.
# The -te option defines the target extent.
warp_cmd = [
    "gdalwarp",
    "-te", str(minX), str(minY), str(maxX), str(maxY),
    "-of", "GTiff",      # Output format
    dem_vrt,             # Input DEM (VRT)
    output_geotiff       # Output cropped GeoTIFF
]

print("Running gdalwarp command:")
print(" ".join(warp_cmd))
subprocess.run(warp_cmd, check=True)
print(f"Cropped DEM saved as: {output_geotiff}")
