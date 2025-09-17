import random
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib as mpl
import numpy as np

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter



# EVENT_STR = "Mississippi_20190617_3AC6_non_flood" #"Mississippi_20190617_42BF_non_flood      unfinished "  #"Mississippi_20190617_9D85_non_flood" #"Mississippi_20190617_5E5F_non_flood"
EVENT_STRS = ["Mississippi_20190617_3AC6_non_flood", "Mississippi_20190617_5E5F_non_flood", "Mississippi_20190617_9D85_non_flood", 
              "Harvey_20170829_D734_non_flood", 
              "Harvey_20170829_3220_non_flood","Florence_20180919_374E_non_flood", "Florence_20180919_B86C_non_flood", 
              "Mississippi_20190617_B9D7_non_flood", "Mississippi_20190617_C310_non_flood"]

# EVENT_STRS = ["Mississippi_20190617_3AC6_non_flood", "Mississippi_20190617_5E5F_non_flood", "Mississippi_20190617_9D85_non_flood", 
#               "Harvey_20170829_D734_non_flood","Harvey_20170831_9366_non_flood" , "Harvey_20170831_0776_non_flood", "Harvey_20170829_B8C4_non_flood", 
#               "Harvey_20170829_3220_non_flood","Florence_20180919_374E_non_flood", "Florence_20180919_B86C_non_flood", 
#               "Mississippi_20190617_B9D7_non_flood", "Mississippi_20190617_C310_non_flood"]
# EVENT_STRS = ["Florence_20180919_374E_non_flood", "Florence_20180919_B86C_non_flood"]

shapefiles = []
# EVENT_STRS =  ["Harvey_20170829_3220_non_flood","Florence_20180919_374E_non_flood", "Florence_20180919_B86C_non_flood", "Harvey_20170829_3220_non_flood",  "Harvey_20170829_B8C4_non_flood.csv  #Harvey_20170831_0776_non_flood.csv #Harvey_20170831_9366_non_flood", "Harvey_20170829_D734_non_flood"
# ,"Mississippi_20190617_B9D7_non_flood", "Mississippi_20190617_B9D7_non_flood" , "Mississippi_20190617_42BF_non_flood", "Mississippi_20190617_3AC6_non_flood" ,"Mississippi_20190617_9D85_non_flood" ,"Mississippi_20190617_5E5F_non_flood"]
for EVENT_STR in EVENT_STRS:
    shapefiles.append(f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/{EVENT_STR}.shp") #'/usr/workspace/lazin1/anaconda_dane/envs/RAPID/flood_img_list_flood_non_flood_test2.csv')  #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_3AC6.csv #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_C310.csv  #/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENT_MISSISSIPPI_non_flood_20190617_B9D7.csv

# shapefiles = [f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/All.shp"]

# List of shapefile paths
# shapefiles = ["shapefile1.shp", "shapefile2.shp", "shapefile3.shp"]

# Load shapefiles and assign random colors
gdfs = [gpd.read_file(shp).to_crs(epsg=3857) for shp in shapefiles]

colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in shapefiles]

# Plot shapefiles with random colors
fig, ax = plt.subplots(figsize=(12, 12))
for gdf, color in zip(gdfs, colors):
    gdf = gdf.to_crs(epsg=4326) #3857  #4326
    proj = gdf.crs
    # gdf['latitude'] = gdf.geometry.centroid.y
    # gdf['longitude'] = gdf.geometry.centroid.x

    # # Print the latitudes and longitudes
    # print("Latitudes:")
    # print(gdf['latitude'])
    # print("longitude:")
    # print(gdf['longitude'])
    gdf.plot(ax=ax, alpha=0.03, edgecolor="k", linewidth=1.5, facecolor=color)


lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=proj, zoom=14) #, zoom=10


# ax.xaxis.get_major_formatter().set_scientific(False)
# ax.yaxis.get_major_formatter().set_scientific(False)
# ax.ticklabel_format(useOffset=False)
plt.title("Multiple Shapefiles with Random Colors")

plt.savefig("/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/flood_events_locations.png")
plt.show()