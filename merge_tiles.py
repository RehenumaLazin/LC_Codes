import glob
import numpy as np
import rasterio
from rasterio.merge import merge

def merge_min(files, output):
    datasets = [rasterio.open(f) for f in files]
    mosaic, transform = merge(datasets, method='min')
    
    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform
    })

    with rasterio.open(output, "w", **out_meta) as dest:
        dest.write(mosaic)

    for ds in datasets:
        ds.close()

# Usage

event=f"/p/vast1/lazin1/UNet_Geotiff_output/Sonoma_event2_prec_dem_LC_Hourly" #f"Sonoma_event2_prec_dem_LC" #f"Sonoma_event2_output_not_normalized_retrain" #"Sonoma_event2_org_output_not_normalized"
#"flood_WM_S1A_IW_GRDH_1SDV_20190617T000326_20190617T000351_027712_0320C9_3AC6"
# Output directory
output_geo_dir=f"/p/vast1/lazin1/UNet_Geotiff_output" #"/p/vast1/lazin1/UNet_Geotiff_output_event_wise_prec_5d_doubled"  #/p/vast1/lazin1/UNet_Geotiff_output_event_wise" #/p/vast1/lazin1/UNet_Geotiff_output" #/p/vast1/lazin1/UNet_Geotiff_output_event_wise

event_dir=f"{output_geo_dir}/{event}"
event_str=f"event2_2019-02-28_Sonoma"
# Define the prefix and directory
prefix=f"{event_str}_crop_"
input_dir=f"{event_dir}" # Replace with the directory containing the GeoTIFF files
output_file=f"{event_dir}/merged_{event_str}.tif"  # Define the output file name

files = sorted(glob.glob(f"{input_dir}/{prefix}*.tif"))
print(f"Found {len(files)} files.")
print(f"Merging into: {output_file}")
print(output_file)
merge_min(files, output_file)
