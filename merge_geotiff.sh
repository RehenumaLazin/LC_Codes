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

# Output directory
output_geo_dir="/p/vast1/lazin1/UNet_Geotiff_output_event_wise_prec_one_fourth_not_30" #/p/vast1/lazin1/UNet_Geotiff_output_event_wise_prec_halved_not_30"  #/p/vast1/lazin1/UNet_Geotiff_outputtest_event_wise_prec_halved_not_30"  #_event_wise_prec_doubled_not_30"  #/p/vast1/lazin1/UNet_Geotiff_output_event_wise" #/p/vast1/lazin1/UNet_Geotiff_output" #/p/vast1/lazin1/UNet_Geotiff_output_event_wise

# Ensure the output directory exists
# mkdir -p "$output_npy_dir"

# Loop through each event string
for event in "${EVENT_STRS[@]}"; do
    # Create a directory for each event inside the output directory
    event_dir="$output_geo_dir/$event"
    # mkdir -p "$event_dir"
    for dir in "$event_dir"/*/; do
        if [ -d "$dir" ]; then
            echo "$dir"
            # if [[ ! "$dir" == */ ]]; then
            #     input_path="$dir/"
            # fi

            # # Extract the last directory name
            last_dir=$(basename "$dir")
            tiff_files=($(find "$dir" -type f -name "*.tif"))
    
            if [ "${#tiff_files[@]}" -gt 0 ]; then
                # Merge all GeoTIFFs in the current directory
                intermediate_output="$dir/merged_$last_dir.tif"
                echo "Merging ${#tiff_files[@]} GeoTIFFs into $intermediate_output..."
                gdal_merge.py -o "$intermediate_output" -of GTiff "${tiff_files[@]}"
                
            fi
        fi
    done
            
done
    # echo "Created directory: $event_dir"

    # Example placeholder action (e.g., touch a placeholder file)
    # touch "$event_dir/placeholder.txt"
        # Collect GeoTIFF files in the current directory


echo "All directories processed."
done