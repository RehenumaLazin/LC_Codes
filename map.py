import cmcrameri.cm as cmc

import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np
# import contextily as ctx
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
# import geopandas as gpd
# import pandas as pd

# Open the raster datasets
TORRENT = f"/p/lustre2/lazin1/DEM/cropped_dem_11467270_extent.tif"
TRITON = f"/p/vast1/lazin1/triton/output_corrected_event2/asc/MH_528_00_WGS84.tif"



points_file = "/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_gauge_height.csv"  # Text file containing X, Y coordinates
points_df = gpd.read_file(points_file) if points_file.endswith(".shp") else pd.read_csv(points_file)

# Keep only relevant columns
points_df = points_df[["STAID", "Lat", "Long"]].copy()

# Rename columns
points_df.rename(columns={"Lat": "Y", "Long": "X"}, inplace=True)

# Check if the file contains headers; if not, assume X, Y structure
if "X" not in points_df.columns or "Y" not in points_df.columns:
    points_df.columns = ["X", "Y"]

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(points_df["X"], points_df["Y"])]

with rasterio.open(TORRENT) as dataset1, rasterio.open(TRITON) as dataset2:
    # Read data
    data1 = dataset1.read(1)  # Read first band
    data2 = dataset2.read(1)
    
    
    data1[data1==0]=np.nan

    
    data2[data2==0]=np.nan

    
    data1 = np.ma.masked_equal(data1, dataset1.nodata)
    data2 = np.ma.masked_equal(data2, dataset2.nodata)
    
    transform1 = dataset1.transform  # Get the transform for projection
    transform2 = dataset2.transform  # Get the transform for projection
    
    crs1 = dataset1.crs  # Get the coordinate system
    crs2 = dataset2.crs  # Get the coordinate system
    
    
    # Calculate extent for correct georeferencing
    extent1 = [transform1.c, transform1.c + dataset1.width * transform1.a,
               transform1.f + dataset1.height * transform1.e, transform1.f]
    
    extent2 = [transform2.c, transform2.c + dataset2.width * transform2.a,
               transform2.f + dataset2.height * transform2.e, transform2.f]

    # Find global min/max for consistent color scale
    vmin = (np.nanmin(data1))
    vmax = (np.nanmax(data1))

    # Create figure
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    # Plot first raster
    img1 = axes.imshow(data1,extent=extent1,  cmap="Greens", vmin=vmin, vmax=vmax)
    # axes[0].set_title("TORRENT")
    # axes.set_aspect("equal")  # Ensure same aspect ratio

    # # Plot second raster
    # img2 = axes[1].imshow(data2, extent=extent2, cmap=cmc.batlow, vmin=vmin, vmax=vmax)
    # # ctx.add_basemap(ax=axes[1], source=ctx.providers.Esri.WorldImagery, crs=crs, alpha=0.3) #, alpha=0.3
    # img2 = axes[1].imshow(data2, extent=extent2, cmap=cmc.batlow, vmin=vmin, vmax=vmax)
    # axes[1].set_title("TRITON")
    # axes[1].set_aspect("equal")  # Ensure same aspect ratio
    

    points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry, crs=crs2)
    
    SONOMA_SHP = "/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/11467270.shp"  # Update with the actual path to the shapefile


    # Read the shapefile
    sonoma_gdf = gpd.read_file(SONOMA_SHP)
    if sonoma_gdf.crs != crs1:
        sonoma_gdf = sonoma_gdf.to_crs(crs1)
        
    # Plot shapefile on both maps
    sonoma_gdf.plot(ax=axes, edgecolor="black", facecolor="none", linewidth=1.2, label="Sonoma Shapefile")
    # sonoma_gdf.plot(ax=axes[1], edgecolor="black", facecolor="none", linewidth=1.2)
    
    
    
    reaches = "/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/18010110/nwm_reaches_18010110.shp"
    
    # Read the shapefile
    sonoma_reaches = gpd.read_file(reaches)
    if sonoma_reaches.crs != crs1:
        sonoma_reaches = sonoma_reaches.to_crs(crs1)
        
    # Plot shapefile on both maps
    


    # Plot points on top
    # points_gdf.plot(ax=axes[1], color="red", marker="o", markersize=20, label="Points")
    
    # Set same x and y ticks for both plots
    axes.set_xticks(np.linspace(*axes.get_xlim(), num=4))
    axes.set_yticks(np.linspace(*axes.get_ylim(), num=6))
    # axes[1].set_xticks(axes[0].get_xticks())
    # axes[1].set_yticks(axes[0].get_yticks())
    
    # Ensure same x/y limits for both subplots
    # axes[1].set_xlim(axes[0].get_xlim())
    # axes[1].set_ylim(axes[0].get_ylim())
    
    # points_gdf.plot(ax=axes[1], color="red", marker="o", markersize=20, label="Points")
    # axes[1].scatter(points_gdf["X"].values, points_gdf["Y"].values, color="red", marker="o", s=20)
    axes.scatter(points_gdf["X"].values, points_gdf["Y"].values, color="red", marker="o", s=20)

    # Add a common colorbar
    cbar = fig.colorbar(img1, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Elevation (m)")
    axes.scatter(points_gdf["X"].values, points_gdf["Y"].values, color="red", marker="o", s=22)
    
    plt.rcParams.update({
    'font.size': 12,  # Increase font size globally
    
    })
    axes.scatter(points_gdf["X"].values, points_gdf["Y"].values, color="red", marker="o", s=22)
    sonoma_reaches.plot(ax=axes, edgecolor="blue", facecolor="none", linewidth=0.5, label="Sonoma Shapefile")
    plt.savefig("/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_streamlines.png")
    plt.show()
