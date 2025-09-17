import geopandas as gpd
import pandas as pd
import glob


events_file = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'
# events_file = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined_reduced_events_files.csv'
combined_df = pd.read_csv(events_file, header=None) 
for idx, raster_path in enumerate(combined_df[0]): #events = [raster_path.split("/")[-1][:-4] 
    event = raster_path.split("/")[-1][:-4]

    output_shapefile = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_waterbody/shapefiles/{event}.shp"  # output shapefile path



    # Load the shapefile into a GeoDataFrame
    gdf = gpd.read_file(output_shapefile)

    # Drop the geometry column to keep only attributes
    attribute_data = gdf.drop(columns="geometry")

    # Save as CSV
    output_shape_csv_file = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_waterbody/shapefiles/{event}.csv"  # output shapefile path
    attribute_data.to_csv(output_shape_csv_file, index=False)
    print("Attribute table saved as attribute_table.csv.")