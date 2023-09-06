#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python 02_train_hps_parallel.py pnet_phaseshuffle2 99
python 02_train_hps_parallel.py pnet_phaseshuffle3 99
python 02_train_hps_parallel.py pnet_magshuffle2 99

