import os
import subprocess
from osgeo import gdal
import glob
import pandas as pd
from multiprocessing import Pool

def process_event(raster_path):
    event = raster_path.split("/")[-1][:-4]
    TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_{event}"
    cropped_outpur_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/cropped_DEM/{event}"

    if os.path.exists(cropped_outpur_dir):
        return f'Exists {cropped_outpur_dir}'
    else:
        os.makedirs(cropped_outpur_dir, exist_ok=True)
        geo_files = glob.glob(f"{TARGET_RASTER_DIR}/*.tif")
        merged_vrt = "/p/lustre1/lazin1/USGS_DEM_10m/merged_USGS_DEM_10m.vrt"

        for geo_file in geo_files:
            low_res_ds = gdal.Open(geo_file)
            min_x, pixel_width, _, max_y, _, pixel_height = low_res_ds.GetGeoTransform()
            rows, cols = low_res_ds.RasterYSize, low_res_ds.RasterXSize
            max_x = min_x + cols * pixel_width
            min_y = max_y + rows * pixel_height
            low_res_projection = low_res_ds.GetProjection()
            low_res_ds = None

            cropped_file = os.path.join(cropped_outpur_dir, 'dem_' + os.path.basename(geo_file))

            subprocess.run([
                "gdalwarp", "-te", str(min_x), str(min_y), str(max_x), str(max_y),
                "-tr", str(pixel_width), str(abs(pixel_height)), "-ts", str(cols), str(rows),
                "-t_srs", low_res_projection, "-r", "nearest", "-of", "GTiff", merged_vrt, cropped_file
            ])

            print(f"Cropped raster saved to {cropped_file}")

    return f"Processed {event}"

if __name__ == "__main__":
    events_file = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv'
    combined_df = pd.read_csv(events_file, header=None)
    raster_paths = combined_df[0].tolist()

    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_event, raster_paths)

    for result in results:
        print(result)
