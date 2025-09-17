#!/bin/bash

#!/bin/bash

#BSUB -G flood

source /p/vast1/lazin1/anaconda3-lassen/etc/profile.d/conda.sh
conda activate /p/vast1/lazin1/anaconda3-lasse/envs/pytorch2.0.1/


# Run with 4 GPUs
# outfile="/usr/workspace/lazin1/anaconda_dane/envs/RAPID/normalized_UNet/reports/Normalized_UNet.out"

python -m torch.distributed.launch --use_env script_LSTM_UNet_NSE.py train

# python -m torch.distributed.launch --use_env --nproc_per_node=4 --nnodes=1 --node_rank=0 script_LSTM_UNet_NSE.py train