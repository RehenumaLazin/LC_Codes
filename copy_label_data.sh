


#!/bin/bash

# Variables
CSV_FILE="/usr/workspace/lazin1/anaconda_dane/envs/RAPID/EVENTS/combined_test.csv"  # Path to the CSV file
SOURCE_DIR="/p/lustre1/lazin1/RAPID_Archive_Flood_Maps"                       # Source directory
DEST_DIR="/p/vast1/lazin1/UNet_inputs/Geotiff_var/WM"                         # Destination directory

# Check if CSV file exists
if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: CSV file '$CSV_FILE' not found."
    exit 1
fi



# Read folder paths from the first column of the CSV and copy them
while IFS=, read -r folder_path _; do
    # Skip empty lines
    [[ -z "$folder_path" ]] && continue

    # Trim leading/trailing whitespace and carriage returns
    folder_path=$(echo "$folder_path" | tr -d '\r' | xargs)

    # Extract folder name from the path
    folder_name=$(basename "$folder_path")
    echo ${folder_name%.*}

    # Define full destination path
    SOURCE_FOLDER="$SOURCE_DIR/Tile_${folder_name%.*}"
    echo $SOURCE_FOLDER
    DEST_FOLDER="$DEST_DIR/Tile_${folder_name%.*}"
    # Create destination directory if it does not exist
    mkdir -p "$DEST_FOLDER"

    # Check if source folder exists
    if [[ -d "$SOURCE_FOLDER" ]]; then
        echo "Copying '$SOURCE_FOLDER' to '$DEST_FOLDER'..."
        cp -r "$SOURCE_FOLDER" "$DEST_FOLDER"
    else
        echo "Warning: Source folder '$folder_path' does not exist. Skipping."
    fi
done < <(tail -n +2 "$CSV_FILE")

echo "Folder copy operation complete."
