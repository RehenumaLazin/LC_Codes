import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio import warp
from netCDF4 import Dataset
from osgeo import gdal
from datetime import datetime, timedelta
import pandas as pd
import glob
import calendar
import xcdat as xc
import xarray as xa



event_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM/Tile_Sonoma/event1/"


files = sorted(glob.glob(f"{event_dir}/*.tif")) #'/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_Threshold_Mississippi_20190617_5E5F_non_flood'
# reference_raster_files= reference_raster_files.to_numpy()


output_geotiff_dir = f"/p/vast1/lazin1/UNet_inputs/Geotiff_var/1D_prec/Sonoma/event1/" #Harvey_D374'  #5E5F
os.makedirs(output_geotiff_dir, exist_ok=True)
path = '/p/lustre2/lazin1/AORC_APCP_surface/'

for f in files:        
    temp_raster_path = os.path.join(output_geotiff_dir, f.split("/")[-1].split("crop")[0][:-1] + '.tif')
    output_geotiff = f"{output_geotiff_dir}/1day_prec_"+ f.split("/")[-1]
    
    
    # temp_raster_path = os.path.join(output_geotiff_dir, reference_raster_file[1].split("crop")[0][:-1]+'.tif')
    # output_geotiff = os.path.join(output_geotiff_dir, '1day_prec_' + reference_raster_file[1])
    reference_raster = f #os.path.join(reference_raster_dir,reference_raster_file[1])
    with rasterio.open(reference_raster) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)
        ref_bounds = ref.bounds

    if not os.path.exists(temp_raster_path):
    
        print(temp_raster_path)
            

        event = f.split("/")[-1].split("_")[0]
        end_date_str = f.split("/")[-1].split("_")[1]
        
        end_ym = end_date_str.split("-")[0] +'_'+ end_date_str.split("-")[1] 
        
        
        start_date_dt = datetime.fromtimestamp(datetime.strptime(end_date_str +" 23" , "%Y-%m-%d %H").timestamp()) - timedelta(days=1) + timedelta(hours=1)
        end_date = datetime.strptime(end_date_str +" 23", "%Y-%m-%d %H")
        start_date = start_date_dt.strftime( "%Y-%m-%d %H")
        print(start_date, end_date)

        
        start_ym = start_date.split("-")[0] + "_" + start_date.split("-")[1]
        end_ym = end_date_str.split("-")[0] +'_'+ end_date_str.split("-")[1] 
        

        
        print(start_ym, end_ym)
        file_pattern = "APCP_surface_*.nc"  # Modify if files are in a different directory
        all_files = sorted(glob.glob(path+file_pattern))  # Get all matching files
        print(len(all_files))
        # Filter files based on year-month
        filtered_files = []
        for file in all_files:
            year = file.split("_")[-2]  
            month = file.split("_")[-1].replace(".nc", "")  # Extract YYYY_MM
            year_month = year + "_" + month
            if start_ym <= year_month <= end_ym:
                filtered_files.append(file)
        print(filtered_files)
        
        # Open and concatenate all selected files
        precip = []
        
        for f in filtered_files:
            precip_slice = xc.open_dataset(f)
            precip_slice = precip_slice.sel(latitude=slice(38.26, 39.5), longitude=slice(-123.5, -122.4))
            precip.append(precip_slice)
        precip = xa.concat(precip, dim='time')
        _, index = np.unique(precip["time"], return_index=True)
        precip = precip.isel(time=index)
        precip = precip.sel(time=slice(start_date, end_date))
        # print(precip)

        # save to tiff:
        data_var = precip['APCP_surface'].fillna(0)
        
        # data_var_flip = data_var.isel(latitude=slice(None, None, -1))  # Flip latitude
        # print('flip',data_var_flip[:1,:5,:1])
        # print('data_var',data_var[:1,:5,:1])
        
        precip_1day= data_var.sum(dim="time")
        # print(data_var_flip.shape, data_var.shape, precip_1day.shape)
        
        transform = from_origin(
        west=precip.longitude.min().values, 
        north=precip.latitude.max().values, 
        xsize=(precip.longitude[1] - precip.longitude[0]).values, 
        ysize=(precip.latitude[1] - precip.latitude[0]).values
        )


        with rasterio.open(
            temp_raster_path,
            "w",
            driver="GTiff",
            height=precip_1day.shape[0],
            width=precip_1day.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform
        ) as temp_dst:
            temp_dst.write((np.flip(precip_1day, axis=0)), 1)

        # Reproject and resample to match the reference raster
        with rasterio.open(temp_raster_path) as src:
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
                # Resample and reproject the data
                warp.reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=warp.Resampling.nearest
                )

        # Clean up temporary file
        import os
        # os.remove(temp_raster_path)

        # Close the NetCDF dataset
        # nc.close()
        
    else:
        with rasterio.open(temp_raster_path) as src:
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
                # Resample and reproject the data
                warp.reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=warp.Resampling.nearest
                )

    # print(f"1-day precipitation GeoTIFF saved at {output_geotiff}")
