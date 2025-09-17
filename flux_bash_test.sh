#!/bin/bash
#flux: -N 4
#flux: -q pbatch
#flux: -B flood
#flux: -t 12h

export MIOPEN_LOG_LEVEL=0

python UNet_normalized_Sonoma_test_loop_output_not_normalize_prec_dem_multiple_events.py test