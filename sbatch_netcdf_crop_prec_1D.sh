#!/bin/bash -l
#SBATCH -N 1
#SBATCH -p pbatch
#SBATCH -t 12:00:00
#SBATCH -o %A-flood.out
#SBATCH -A flood

source /g/g92/lazin1/anaconda/etc/profile.d/conda.sh
conda activate /usr/workspace/lazin1/anaconda_dane/envs/RAPID/

python  netcdf_to_cropped_geotiff_1day_prec_sonoma.py &
wait