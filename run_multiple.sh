#!/bin/bash

#BSUB -G flood
#BSUB -nnodes 10
#BSUB -q pbatch

source /p/vast1/lazin1/anaconda3-lassen/etc/profile.d/conda.sh
conda activate /p/vast1/lazin1/anaconda3-lasse/envs/pytorch2.0.1/


python -m torch.distributed.launch --use_env UNet_normalized_Sonoma_test_loop_output_not_normalize_prec_dem_multiple_events.py train