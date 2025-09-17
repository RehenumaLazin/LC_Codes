import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Path to your multi-band GeoTIFF ---
tiff_path = "/p/lustre2/lazin1/IMERG/precip_IMERG_GuadalupeRiver.tif"

# --- Define start time ---
start_time = pd.Timestamp("2025-07-01 00:00")  # Adjust if needed

# --- Read data and compute mean per band ---
with rasterio.open(tiff_path) as src:
    num_bands = src.count
    mean_series = []

    for band in range(1, num_bands + 1):
        data = src.read(band).astype(np.float32)

        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)

        mean_value = np.nanmean(data)
        mean_series.append(mean_value)

# --- Create datetime index for 3-hour steps ---
time_index = pd.date_range(start=start_time, periods=num_bands, freq="3H")

# --- Plotting ---
plt.figure(figsize=(12, 5))
plt.plot(time_index, mean_series, marker='o', linestyle='-')
plt.title("Mean Precipitation Time Series (3-hourly)")
plt.xlabel("Time")
plt.ylabel("Mean Value")
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
