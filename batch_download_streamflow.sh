#!/bin/bash
#SBATCH -p pbatch               # cluster
#SBATCH -N 16
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-core 10
#SBATCH -A flood
#SBATCH --exclusive
#SBATCH -o run.log
#SBATCH -e error.log




source /p/vast1/lazin1/anaconda3-lassen/etc/profile.d/conda.sh
conda activate /usr/workspace/lazin1/anaconda_dane/envs/RAPID
cd /usr/workspace/lazin1/anaconda_dane/envs/RAPID/Codes

# Run with 4 GPUs
# outfile="/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/reports/Normalized_UNet.out"

srun -N 16 -n 32  python Download_NOAA_streamflow.py 




