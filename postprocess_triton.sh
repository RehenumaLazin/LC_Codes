#!/bin/bash

#BSUB -G flood
source /g/g92/lazin1/anaconda/etc/profile.d/conda.sh

# Set up directories
EVENT=2
BIN_DIR="/p/vast1/lazin1/triton/output_runoff_event"$EVENT"_FEb22/bin/"
TEST_DIR="$BIN_DIR/test/"
ASC_DIR="/p/vast1/lazin1/triton/output_runoff_event"$EVENT"_FEb22/asc/"
BIN2ASCII_DIR="/p/vast1/lazin1/triton/tools/bin2ascii/"
CONDA_ENV="/p/vast1/lazin1/anaconda3-lasse/envs/geo"
HEADER_FILE="/p/vast1/lazin1/triton/input/dem/asc/case_sonoma.header"

# Create necessary directories
mkdir -p "$TEST_DIR"
mkdir -p "$ASC_DIR"

# Copy files with 48 interval steps
for i in $(seq 0 1 9999); do
    echo $i
    FILE=$(printf "H_%02d_00.out" "$i")
    if [[ -f "$BIN_DIR/$FILE" ]]; then
        cp "$BIN_DIR/$FILE" "$TEST_DIR/"
        i_last=$i
    fi
done

echo $i_last
MH_FILE=$(printf "MH_%d_00.out" "$i_last")
cp "$BIN_DIR/$MH_FILE" "$TEST_DIR/"

# Navigate to bin2ascii directory
cd "$BIN2ASCII_DIR" || exit 1
echo $(pwd)


# Load GCC module
module load gcc

# Compile bin2ascii
make clean && make

# Convert binary to ASCII
./bin2ascii "$TEST_DIR" "$ASC_DIR" 2

# Move to ASC directory
cd "$ASC_DIR" || exit 1
echo $(pwd)
conda activate /p/vast1/lazin1/anaconda3-lasse/envs/geo/ 
# Activate Conda environment
# source /p/vast1/lazin1/anaconda3-lasse/etc/profile.d/conda.sh
# conda init
# conda activate "$CONDA_ENV"
# Convert .out files to .asc with headers
for FILE in H_*_00.out MH_*_00.out; do
    ASC_FILE="${FILE/.out/.asc}"
    echo $ASC_FILE
    cat "$HEADER_FILE" "$FILE" > "$ASC_FILE"
    
    # Convert to GeoTIFF
    TIF_FILE="${ASC_FILE/.asc/.tif}"
    gdal_translate -of GTiff -a_srs EPSG:32610 "$ASC_FILE" "$TIF_FILE"
done

echo "Processing complete. GeoTIFF files are in $ASC_DIR."
