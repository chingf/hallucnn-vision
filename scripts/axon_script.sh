#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

python 02_train_hps_parallel.py pnet3 99
python 02_train_hps_parallel.py pnet3 99
python 02_train_hps_parallel.py pnet3 99
python 02_train_hps_parallel.py pnet_noisy3 99
python 02_train_hps_parallel.py pnet_noisy3 99
python 02_train_hps_parallel.py pnet_noisy3 99
python 02_train_hps_parallel.py pnet_magshuffle3 99
python 02_train_hps_parallel.py pnet_magshuffle3 99
python 02_train_hps_parallel.py pnet_magshuffle3 99

