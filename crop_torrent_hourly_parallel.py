import os
import numpy as np
import rasterio
from rasterio import warp
from datetime import datetime, timedelta
import pandas as pd
import glob
from multiprocessing import Pool, cpu_count


def process_event(idx_s_e):
    idx, s, e, s_short, e_short = idx_s_e

    

    start_time = datetime.strptime(s_short + " 00", "%Y-%m-%d %H")
    end_time = datetime.strptime(e_short + " 23", "%Y-%m-%d %H")

    start_dt = datetime.strptime(s + " 00", "%Y-%m-%d %H")
    end_dt = datetime.strptime(e, "%Y-%m-%d")
    date_obj = datetime.strptime(s, "%Y-%m-%d")

    event_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma_hourly/event{idx+1}"
    os.makedirs(event_dir, exist_ok=True)

    geo_files = glob.glob(f"{TARGET_RASTER_DIR}/*.tif")
    temp_raster_dir = f"/p/lustre1/lazin1/flood/Sonoma/RussianRiver_{s}_{e}"

    if not os.path.exists(temp_raster_dir):
        return

    files = sorted(glob.glob(os.path.join(temp_raster_dir, "depth-1-*.tif")), key=os.path.getmtime)
    for d, file in enumerate(files):
        try:
            minutes = int(os.path.basename(file).split("-")[-1].split(".")[0])
        except ValueError:
            continue

        timestamp = start_time + timedelta(minutes=minutes)
        if timestamp > end_time:
            continue

        date = timestamp.strftime('%Y%m%d%H')
        os.makedirs(f"{event_dir}/{date}", exist_ok=True)

        for reference_raster_file in geo_files:
            f = os.path.basename(reference_raster_file)
            output_geotiff = f"{event_dir}/event{idx+1}_{date}_{f}"

            if os.path.exists(output_geotiff):
                continue

            with rasterio.open(reference_raster_file) as ref:
                ref_transform = ref.transform
                ref_crs = ref.crs
                ref_shape = (ref.height, ref.width)

            with rasterio.open(file) as src:
                with rasterio.open(
                    output_geotiff,
                    "w",
                    driver="GTiff",
                    height=ref_shape[0],
                    width=ref_shape[1],
                    count=1,
                    dtype="float32",
                    crs=ref_crs,
                    transform=ref_transform,
                    nodata=-9999
                ) as dst:
                    warp.reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=warp.Resampling.nearest
                    )


# Load data
events_file = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_events_short.csv"
TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_Sonoma"

combined_df = pd.read_csv(events_file)
start_dates = combined_df['start'].to_numpy()
end_dates = combined_df['end'].to_numpy()

events_all = f"/usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes/Sonoma_shapefile/Sonoma_events.csv"
combined_df_all = pd.read_csv(events_all)
start_dates_all = combined_df_all['start'].to_numpy()
end_dates_all = combined_df_all['end'].to_numpy()

# Create list of input tuples for each event
inputs = [
    (idx, s, e, start_dates[idx], end_dates[idx])
    for idx, (s, e) in enumerate(zip(start_dates_all, end_dates_all))
    if idx in range(6)
]

# Use multiprocessing pool
if __name__ == '__main__':
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_event, inputs)
