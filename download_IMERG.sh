#!/bin/bash

# Script to download GPM IMERG-HH data using date-based URLs

# Define Earthdata credentials
USER="rehenuma.lazin@uconn.edu"
PASSWORD="Umayer.uzayer0120"

# Prompt user for start and end dates
read -p "Enter the start date (YYYY-MM-DD): " START_DATE
read -p "Enter the end date (YYYY-MM-DD): " END_DATE

# Validate input dates
if ! date -d "$START_DATE" &>/dev/null; then
    echo "❌ Invalid start date: $START_DATE"
    exit 1
fi

if ! date -d "$END_DATE" &>/dev/null; then
    echo "❌ Invalid end date: $END_DATE"
    exit 1
fi

# Convert to epoch seconds
START_SEC=$(date -d "$START_DATE" +%s)
END_SEC=$(date -d "$END_DATE" +%s)

# Check logical order
if [ "$START_SEC" -gt "$END_SEC" ]; then
    echo "❌ Start date must be earlier than end date."
    exit 1
fi

# Base URL
BASE_URL="https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHL.07/"

# Loop through each day
CURRENT_SEC=$START_SEC
while [ "$CURRENT_SEC" -le "$END_SEC" ]; do
    # Extract date components
    YEAR=$(date -d "@$CURRENT_SEC" +%Y)
    MONTH=$(date -d "@$CURRENT_SEC" +%m)
    DAY=$(date -d "@$CURRENT_SEC" +%d)
    DATE_STR=$(date -d "@$CURRENT_SEC" +%Y-%m-%d)

    # Calculate the Day of Year (DOY)
    DOY=$(date -d "$DATE_STR" +%j)

    # Print date info
    echo "📆 Date: $DATE_STR | DOY: $DOY"

    # Construct URL and output directory
    URL="${BASE_URL}${YEAR}/${DOY}/"
    OUTPUT_DIR="${YEAR}_${MONTH}_${DAY}"
    mkdir -p "$OUTPUT_DIR"

    echo "📥 Downloading from: $URL"

    # Use wget to fetch data
    wget --user="$USER" --password="$PASSWORD" -r -np -nH --cut-dirs=5 \
         --reject "index.html*" "$URL" -P "$OUTPUT_DIR"

    # Increment by 1 day
    CURRENT_SEC=$((CURRENT_SEC + 86400))
done

echo "✅ Download complete."
