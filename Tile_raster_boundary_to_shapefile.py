import os
import rasterio
from rasterio.features import geometry_window
from shapely.geometry import box, mapping
import fiona
from fiona.crs import from_epsg
import pandas as pd
import glob

# Function to get the bounding box of a raster
def get_raster_boundary(raster_path):
    with rasterio.open(raster_path) as src:
        
        bounds = src.bounds  # Get the bounds (left, bottom, right, top)
        crs = src.crs  # Get the coordinate reference system
        boundary_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)  # Create shapely box
        return boundary_geom, crs

# Function to save boundaries as shapefile
def save_boundaries_as_shapefile(raster_paths, shapefile_path):
    schema = {
        'geometry': 'Polygon',  # We're saving polygons (bounding boxes)
        'properties': {
            'id': 'int',
            'ras_name': 'str',
            'start': 'str',
            'end': 'str'
            },  # Add an ID to each feature
    }
    
    with fiona.open(shapefile_path, 'w', driver='ESRI Shapefile', crs=from_epsg(4326), schema=schema) as shp:
        for idx, raster_path in enumerate(raster_paths):
            try:
                # print(f"Processing {raster_path}...")
                # print(f"Processing {raster_path}...")
                boundary_geom, crs = get_raster_boundary(raster_path)
                print(boundary_geom, idx,raster_path.split("/")[-1], raster_path.split("/")[-1].split("_")[6], raster_path.split("/")[-1].split("_")[7] )
                # Write the geometry and properties to the shapefile
                shp.write({
                    'geometry': mapping(boundary_geom),
                    'properties': {
                        'id': idx,
                        'ras_name': raster_path.split("/")[-1],
                        'start': raster_path.split("/")[-1].split("_")[6],
                        'end': raster_path.split("/")[-1].split("_")[7]
                        },
                    'ras_name': raster_path.split("/")[-1].split("_")[5]
                })
                print(f"Boundary for {raster_path} saved as feature {idx}")
            except:
                continue

# List of raster file paths
# raster_paths = ['raster1.tif', 'raster2.tif', 'raster3.tif']  # Replace with your actual file paths

# raster_paths = pd.read_csv('/usr/WS1/lazin1/anaconda_dane/envs/RAPID/flood_image_list.csv')
# raster_paths= raster_paths.to_numpy()


# EVENT_STR = "Mississippi_20190617_C310_non_flood" #"Mississippi_20190617_B9D7_non_flood"  # "Mississippi_20190617_42BF_non_flood"  #"Mississippi_20190617_3AC6_non_flood" #"Mississippi_20190617_9D85_non_flood" #"Mississippi_20190617_5E5F_non_flood"

EVENT_STR = "All"
TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_{EVENT_STR}" #'/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood'  # Reference raster for extent and resolution




# Define the path to the csv containing your target GeoTIFF files
# TILE_LIST_CSV = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Mississippi_20190617_9D85_non_flood.csv"   #"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Mississippi_20190617_5E5F_non_flood.csv"


# raster_paths = glob.glob('/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_Flood_Images/*')
raster_paths = glob.glob(f"{TARGET_RASTER_DIR}/*") #/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood/*')
# for idx, raster_path in enumerate(raster_paths):
#     print(f"test  {raster_path[0]}...")
# output_shapefile = '/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_Flood_imgs_shapefile.shp'  # output shapefile path
output_shapefile = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/{EVENT_STR}.shp"  # output shapefile path

# Output shapefile path
# output_shapefile = 'raster_boundaries.shp'

# Save raster boundaries to shapefile
save_boundaries_as_shapefile(raster_paths, output_shapefile)
