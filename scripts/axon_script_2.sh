#!/bin/bash 

trap "exit" INT TERM ERR
trap "kill 0" EXIT

#python 03_save_validation_activations.py pnet2 99
#python 03_save_validation_activations.py pnet3 99
#python 03_save_validation_activations.py pnet_noisy 99
#python 03_save_validation_activations.py pnet_noisy2 99
python 03_save_validation_activations.py pnet_noisy3 99
python 03_save_validation_activations.py pnet_phaseshuffle 99
python 03_save_validation_activations.py pnet_magshuffle 99
python 03_save_validation_activations.py pnet_magshuffle3 99
