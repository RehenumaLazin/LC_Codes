

#!/bin/bash

#BSUB -G flood
#BSUB -nnodes 10
#BSUB -q pbatch

source /p/vast1/lazin1/anaconda3-lassen/etc/profile.d/conda.sh
conda activate /p/vast1/lazin1/anaconda3-lasse/envs/pytorch2.0.1/



python -m torch.distributed.launch --use_env script.py train
# python -m torch.distributed.launch --use_env --nproc_per_node=4 --nnodes=1 --node_rank=0 script.py train
