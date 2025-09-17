#!/bin/bash
#!/bin/bash
#flux: -N 4
#flux: -q pbatch
#flux: -B flood
#flux: -t 12h

export MIOPEN_LOG_LEVEL=0

python script_normalized.py train
#UNet_normalized_Sonoma_test_loop_output_not_normalize_prec_dem_LC_multiple_events_Attention.py