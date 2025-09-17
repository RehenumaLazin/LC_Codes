import os
import subprocess

event = f"sonoma_county_11467270"
# cropped_outpur_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/{event}"
cropped_outpur_dir = f"/p/lustre2/lazin1/DEM"
merged_vrt = "/p/lustre1/lazin1/USGS_DEM_10m/merged_USGS_DEM_10m.vrt"  # The merged VRT file
shapefile_list = ["/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/11467270.shp"]

# Step 2: Crop the merged VRT using each shapefile as a cutline
for i, shp in enumerate(shapefile_list, start=1):
    cropped_file = os.path.join(cropped_outpur_dir, f"cropped_dem_11467270_test2.tif")
    
    # Build the gdalwarp command with -cutline and -crop_to_cutline options
    warp_command = [
        "gdalwarp",
        "-cutline", shp,         # Use the shapefile as the cutline
        "-crop_to_cutline",       # Crop the DEM to the exact shape geometry
        "-of", "GTiff",           # Output format as GeoTIFF
        merged_vrt,               # Input VRT file
        cropped_file              # Output cropped file
    ]
    
    print(f"Cropping with shapefile: {shp}")
    print("Running command:")
    print(" ".join(warp_command))
    subprocess.run(warp_command, check=True)
    print(f"Cropped DEM saved to: {cropped_file}")

print("Processing complete!")
