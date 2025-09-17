import s3fs
import xarray as xr
import os
import calendar
import datetime
from multiprocessing import Pool, cpu_count

def process_day(args):
    y, m, d = args
    fs = s3fs.S3FileSystem(anon=True)  # Anonymous access for NOAA data
    output_dir = f"/p/lustre2/lazin1/NOAA_streamflow_WRF_Hydro/{y}"
    os.makedirs(output_dir, exist_ok=True)

    date = datetime.datetime(y, m, d).strftime("%Y%m%d")
    base_s3_path = f"s3://noaa-nwm-retrospective-3-0-pds/CONUS/netcdf/CHRTOUT/{y}/"
    file_times = [f"{date}{str(hour).zfill(2)}00" for hour in range(0, 24, 1)]
    file_paths = [f"{base_s3_path}{time}.CHRTOUT_DOMAIN1" for time in file_times]

    datasets = []
    for file_path in file_paths:
        try:
            print(f"Accessing: {file_path}")
            with fs.open(file_path, 'rb') as f:
                ds = xr.open_dataset(f, engine='h5netcdf')[["streamflow"]].load()
                datasets.append(ds)
        except Exception as e:
            print(f"Error accessing {file_path}: {e}")

    if datasets:
        combined_dataset = xr.concat(datasets, dim="time")
        daily_average = combined_dataset.mean(dim='time')
        output_file = f"{output_dir}/{date}.nc"
        daily_average.to_netcdf(output_file)
        print(f"Combined NetCDF file saved as: {output_file}")
    else:
        print(f"No data available for {date}")

if __name__ == "__main__":
    years = range(2001, 2025)
    all_dates = [(y, m, d) for y in years for m in range(1, 13) for d in range(1, calendar.monthrange(y, m)[1] + 1)]

    with Pool(processes=cpu_count()) as pool:
        pool.map(process_day, all_dates)
