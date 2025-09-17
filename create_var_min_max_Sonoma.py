import rasterio
import numpy as np
import glob
import os

# Path to your hourly precipitation GeoTIFF files

event = f"event3"
precip_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/1H_prec/Sonoma/{event}/"   # e.g., contains files like 2019022701_patch01.tif
tif_files = sorted(glob.glob(os.path.join(precip_dir, f"{event}_*_Sonoma.tif")))

all_mins = []
all_maxs = []

for f in tif_files:
    print(f)
    with rasterio.open(f) as src:
        arr = src.read(1).astype(np.float32)  # Read the first band
        arr = arr[arr != src.nodata]          # Remove NoData values if present
        if arr.size > 0:                      # Avoid empty arrays
            all_mins.append(np.min(arr))
            all_maxs.append(np.max(arr))

global_min = np.min(all_mins)
global_max = np.max(all_maxs)

print(f"📉 Minimum precipitation in dataset: {global_min:.3f}")
print(f"📈 Maximum precipitation in dataset: {global_max:.3f}")
