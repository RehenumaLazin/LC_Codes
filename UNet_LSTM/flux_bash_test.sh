#!/bin/bash
#flux: -N 4
#flux: -q pbatch
#flux: -B flood
#flux: -t 12h

export MIOPEN_LOG_LEVEL=0

python script.py test