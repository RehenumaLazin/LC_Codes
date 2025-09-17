#!/bin/bash

# Define the list of event strings #"Mississippi_20190617_9D85_non_flood"
EVENT_STRS=(
    "Mississippi_20190617_5E5F_non_flood"
    "Mississippi_20190617_3AC6_non_flood"
    "Harvey_20170829_D734_non_flood"
    "Harvey_20170831_9366_non_flood"
    "Harvey_20170831_0776_non_flood"
    "Harvey_20170829_B8C4_non_flood"
    "Harvey_20170829_3220_non_flood"
    "Florence_20180919_374E_non_flood"
    "Florence_20180919_B86C_non_flood"
    "Mississippi_20190617_C310_non_flood"
)

event="Sonoma_event2_org_output_not_normalized"
#"flood_WM_S1A_IW_GRDH_1SDV_20190617T000326_20190617T000351_027712_0320C9_3AC6"
# Output directory
output_geo_dir="/p/vast1/lazin1/UNet_Geotiff_output" #"/p/vast1/lazin1/UNet_Geotiff_output_event_wise_prec_5d_doubled"  #/p/vast1/lazin1/UNet_Geotiff_output_event_wise" #/p/vast1/lazin1/UNet_Geotiff_output" #/p/vast1/lazin1/UNet_Geotiff_output_event_wise

# Ensure the output directory exists
# mkdir -p "$output_npy_dir"

# Loop through each event string
# for event in "${EVENT_STRS[@]}"; do
    # Create a directory for each event inside the output directory
    #/p/vast1/lazin1/UNet_Geotiff_output/flood_WM_S1A_IW_GRDH_1SDV_20190617T000441_20190617T000506_027712_0320C9_42BF
event_dir="$output_geo_dir/$event"
#!/bin/bash

# Ensure GDAL is installed
if ! command -v gdal_merge.py &> /dev/null; then
    echo "Error: gdal_merge.py is not installed. Please install GDAL."
    exit 1
fi
event_str="event2_2019-02-26_Sonoma"
# Define the prefix and directory
prefix="${event_str}_crop_"
input_dir=$event_dir  # Replace with the directory containing the GeoTIFF files
output_file="$event_dir/merged_${event_str}.tif"  # Define the output file name

# Find all GeoTIFF files that start with the specified prefix
tiff_files=($(find "$input_dir" -type f -name "${prefix}*.tif"))

# Check if any matching files are found
if [ "${#tiff_files[@]}" -eq 0 ]; then
    echo "Error: No GeoTIFF files found with prefix '$prefix' in directory '$input_dir'."
    exit 1
fi

# Print the files to be merged
echo "Found ${#tiff_files[@]} GeoTIFF files to merge:"
for file in "${tiff_files[@]}"; do
    echo "  $file"
done

# Merge the GeoTIFF files
echo "Merging GeoTIFF files into $output_file..."
gdal_merge.py -o "$output_file" -of GTiff "${tiff_files[@]}"

# gdalwarp -overwrite -r bilinear \
#   -srcnodata 0 -dstnodata 0 \
#   "${tiff_files[@]}" "$output_file"

# Check if the merge was successful
if [ $? -eq 0 ]; then
    echo "Merge completed successfully. Output file: $output_file"
else
    echo "Error: Failed to merge GeoTIFF files."
    exit 1
fi
