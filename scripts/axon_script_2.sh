#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python 02_flexible.py pnet_noisy2 99
python 02_flexible.py pnet_magshuffle2 99
python 02_flexible.py pnet_phaseshuffle2 99
