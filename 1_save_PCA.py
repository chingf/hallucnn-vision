import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import seaborn as sns
import datetime
from scipy.stats import sem
import matplotlib.cm as cm
import pathlib
import traceback
import gc

##### ARGS ######
netname = str(sys.argv[1]) # pnet
if len(sys.argv) > 2 and str(sys.argv[2]) == 'shufflehalf':
    print('Shuffle-half PCA. Clean even + noisy odd for model fit.')
    shufflehalf = True
else:
    shufflehalf = False
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn_vision/'
dataset_root = f'{engram_dir}imagenet/'
ckpt_root = f'{engram_dir}checkpoints/'
hps_root = f'{engram_dir}hyperparams/'
activations_dir = f'{engram_dir}activations/{netname}/'
pca_activations_dir = f'{engram_dir}activations_pca/{netname}/'

##### HELPER FUNCTIONS #####
def get_data_and_fit_PCA(conv_idx, t, pca_activations_dir):
    clean_activ_dir = f'{activations_dir}none_lvl_0/'
    for results_file in os.listdir(clean_activ_dir):
        results_filepath = f'{clean_activ_dir}{results_file}'
        results = h5py.File(results_filepath, 'r')
    clean_activ = np.array(results[f'{conv_idx}_{t}_activations'])
    n_data = clean_activ.shape[0]
    clean_activ = clean_activ.reshape((n_data, -1))
    print(f'Runing PCA on clean with data shape {clean_activ.shape}')
    pca = PCA()
    pca.fit(clean_activ)
    pca_filename = f'PCA_clean_conv{conv_idx}_t{t}'
    with open(f'{pca_activations_dir}{pca_filename}.p', 'wb') as f:
        pickle.dump(pca, f, protocol=4)   

    for noise_type in os.listdir(activations_dir):
        if 'lvl' not in noise_type: continue
        if 'none' in noise_type: continue
        activ_dir = f'{activations_dir}{noise_type}/'
        for results_file in os.listdir(activ_dir):
            results_filepath = f'{activ_dir}{results_file}'
            results = h5py.File(results_filepath, 'r')
        activ = np.array(results[f'{conv_idx}_{t}_activations'])
        n_data = activ.shape[0]
        activ = activ.reshape((n_data, -1))
        print(f'Runing PCA on {noise_type} with data shape {activ.shape}')
        if shufflehalf:
            _activ = np.zeros(activ.shape)
            _activ[::2] = np.copy(clean_activ[::2])
            _activ[1::2] = np.copy(activ[1::2])
            odds_activ = activ[1::2]
            pca_filename = f'PCA_{noise_type}_shufflehalf_conv{conv_idx}_t{t}'
        else:
            _activ = activ
            pca_filename = f'PCA_{noise_type}_conv{conv_idx}_t{t}'
        pca = PCA()
        pca.fit(_activ)
        with open(f'{pca_activations_dir}{pca_filename}.p', 'wb') as f:
            pickle.dump(pca, f, protocol=4)   


def get_cpu_usage():
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t --giga').readlines()[-1].split()[1:])

    # Memory usage
    p_used = round((used_memory/total_memory) * 100, 2)
    print(f"RAM {used_memory} GB, {p_used}% used")

##### MAIN CALL #####

if __name__ == "__main__":
    os.makedirs(pca_activations_dir, exist_ok=True)
    for conv_idx in [1, 2, 3, 4, 5, 6, 7, 8]:
        for t in [0, 1, 2, 3, 4]:
            print(f'====== PROCESSING LAYER {conv_idx}, TIMESTEP {t} ======')
            get_data_and_fit_PCA(conv_idx, t, pca_activations_dir)
            gc.collect()
