#!/bin/bash
#SBATCH -p pbatch               # cluster
#SBATCH -t 12:00:00
#SBATCH -A flood
#SBATCH --exclusive
#SBATCH -o run_test.log
#SBATCH -e error_test.log




source /p/vast1/lazin1/anaconda3-lassen/etc/profile.d/conda.sh
conda activate /usr/workspace/lazin1/anaconda_dane/envs/RAPID
cd /usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes

# Run with 4 GPUs
# outfile="/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/reports/Normalized_UNet.out"

srun  python Download_NOAA_streamflow.py 




