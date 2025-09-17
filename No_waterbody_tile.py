import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio import warp
import pandas as pd
from osgeo import gdal



def crop_geotiff(input_tiff_path, output_dir, crop_width, crop_height, output_geotiff):
    # Open the large GeoTIFF image
    ds = gdal.Open(input_tiff_path)
    gt = ds.GetGeoTransform()
    projection = ds.GetProjection()
    band = ds.GetRasterBand(1)
    
    # Get the data type of the raster band
    data_type = band.DataType

    # Translate GDAL data type into a human-readable string
    data_type_name = gdal.GetDataTypeName(data_type)
    # print(data_type_name,data_type )
    No_data = band.GetNoDataValue()
    arr = np.array(band.ReadAsArray())
    # print('Array', np.max(arr), np.min(arr), No_data)
    
    ds_LC = gdal.Open(output_geotiff)
    band_LC = ds_LC.GetRasterBand(1)
    arr_LC = np.array(band_LC.ReadAsArray())
    ds_LC = None
    
    
    arr=np.where(arr_LC == 11, 0, arr) #arr[arr_LC==11] = 0.0
    arr = np.where(arr == No_data, np.nan, arr)
    # Create a new raster file to save the modified band
    output_tiff_path = os.path.join(output_dir,'target_no_waterbody.tif')
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_tiff_path,
        ds.RasterXSize,
        ds.RasterYSize,
        1,  # Number of bands
        data_type,  # Data type
    )

    # Set the GeoTransform and Projection for the output file
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(projection)

    # Write the modified data to the new file
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(arr)
    out_band.SetNoDataValue(255)
    
    out_band.FlushCache()
    out_ds = None
    # print('arr nanmax',np.nanmax(arr), 'arr_No_data', No_data)

    ds = None
    
    ds = gdal.Open(output_tiff_path)
    gt = ds.GetGeoTransform()
    projection = ds.GetProjection()
    band = ds.GetRasterBand(1)
    check_array = band.ReadAsArray()
    No_data_check = band.GetNoDataValue()
    data_type_check = band.DataType
    # print('Check_array',np.nanmax(check_array),'n0_data',No_data_check, 'data_type_check', data_type_check)
    
    arr=np.where(arr == No_data, np.nan, arr) #arr[arr==No_data ] = np.nan
    # print('Array', np.nanmax(arr), np.min(arr), No_data)
    img_width = ds.RasterXSize
    img_height = ds.RasterYSize

    # Calculate number of tiles in x and y directions
    x_tiles = img_width // crop_width
    y_tiles = img_height // crop_height
    count = 1
    tile_data = np.array(band.ReadAsArray())
    # print(x_tiles, y_tiles, print(np.sum(tile_data)), tile_data)
    check=0
    for i in range(x_tiles):
        for j in range(y_tiles):
            # Calculate pixel coordinates for the tile
            offset_x = i * crop_width
            offset_y = j * crop_height
            
            # Read the data for the tile
            tile_data = (band.ReadAsArray(offset_x, offset_y, crop_width, crop_height))
            check +=1
            # print(np.sum(tile_data),check)
            tile_data=np.where(tile_data == No_data, np.nan, tile_data)
            
            
            # if np.sum(np.isnan(tile_data))>0.0 or np.sum(tile_data)==0.0: #(np.sum(tile_data)/(crop_width*crop_height))<0.00001
            #     # print(np.sum(tile_data)/(crop_width*crop_height))
            #     # print(np.max(tile_data))
            #     continue
            # else:
            if np.sum(tile_data)>0.0 and (np.sum(tile_data)/(crop_width*crop_height))>0.01:
                print(np.max(tile_data), np.min(tile_data))

            
                # Create the cropped output file
                driver = gdal.GetDriverByName('GTiff')
                output_file = os.path.join(output_dir, raster_path.split("/")[-1][:-4]+f'_crop_{count}.tif')
                out_ds = driver.Create(output_file, crop_width, crop_height, 1, band.DataType)
                
                # Set GeoTransform and Projection for the cropped file
                new_gt = (
                    gt[0] + offset_x * gt[1], gt[1], gt[2],
                    gt[3] + offset_y * gt[5], gt[4], gt[5]
                )
                out_ds.SetGeoTransform(new_gt)
                out_ds.SetProjection(ds.GetProjection())
                
                
                # Write the cropped data to the output file
                out_ds.GetRasterBand(1).WriteArray(tile_data)
                # out_ds.SetNoDataValue(No_data)
                out_ds.FlushCache()
                
                # Clean up
                out_ds = None
                count += 1
                
    # Flush and close the dataset
    ds = None
    # os.remove(output_tiff_path)

crop_width = 512                       # Width of each cropped image (in pixels)
crop_height = 512  

# Input files and directories
events_file = '/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined.csv' #combined.csv'
combined_df = pd.read_csv(events_file, header=None)

for idx, raster_path in enumerate(combined_df[0]): 
    event = raster_path.split("/")[-1][:-4]  # Extract event name from raster path
    TARGET_RASTER_DIR = f"/p/lustre1/lazin1/RAPID_Archive_Flood_Maps/Tile_No_waterbody/{event}"
    
    if os.path.exists(TARGET_RASTER_DIR):
        print('Exists', TARGET_RASTER_DIR)
        # continue
    else:
        print('New', TARGET_RASTER_DIR)
        os.makedirs(TARGET_RASTER_DIR, exist_ok=True)

    # Output directory for cropped raster
    cropped_output_dir = TARGET_RASTER_DIR
    temp_raster_path = f"/p/lustre2/lazin1/Annual_NLCD_LndCov_2023_CU_C1V0.tif"
    output_geotiff = os.path.join(cropped_output_dir, 'LC_' + event + '.tif')
    reference_raster = raster_path

    # Process the reference raster
    with rasterio.open(reference_raster) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)
        ref_bounds = ref.bounds
        ref_nodata = ref.nodata  # Get NoData value from reference raster

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
            nodata=ref_nodata  # Use the same NoData value as the reference raster
        ) as dst:
            # Create an array for the destination
            destination_array = np.empty((ref_shape[0], ref_shape[1]), dtype="float32")
            destination_array.fill(ref_nodata)  # Initialize with NoData value
            

            # Reproject the data
            warp.reproject(
                source=rasterio.band(src, 1),
                destination=destination_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=warp.Resampling.nearest
            )

            # Mask with NoData where reference has NoData
            with rasterio.open(reference_raster) as ref:
                reference_data = ref.read(1)  # Read reference raster band
                destination_array[reference_data == ref_nodata] = ref_nodata  # Apply NoData mask
                # destination_array[destination_array == 11] = ref_nodata

            # Write to the output raster
            dst.write(destination_array, 1)


    print(f"Processed and saved: {output_geotiff}")
    try:
        print(f"Processing {raster_path}...")
        crop_geotiff(raster_path, TARGET_RASTER_DIR, crop_width, crop_height, output_geotiff)
    except:
        continue
    
    os.remove(output_geotiff)
