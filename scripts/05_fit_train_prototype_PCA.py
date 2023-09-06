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
if len(sys.argv) > 2:
    shuffle_seed = int(sys.argv[2])
    print(f'Shuffle-label PCA with seed {shuffle_seed}')
    shuffle = True
else:
    shuffle = False
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn_vision_resnet/'
dataset_root = '/mnt/smb/locker/abbott-locker/hcnn_vision/imagenet/'
ckpt_root = f'{engram_dir}checkpoints/'
hps_root = f'{engram_dir}hyperparams/'
activations_dir = f'{engram_dir}train_activations/{netname}/'
pca_activations_dir = f'{engram_dir}train_prototype_PCA/{netname}/'

##### HELPER FUNCTIONS #####
n_units_per_layer = {
    1: (64, 112, 112), 2: (64, 56, 56), 3: (128, 28, 28),
    4: (256, 14, 14), 5: (512, 7, 7)}
n_labels = 1000

def get_data_and_fit_PCA(conv_idx, t, pca_activations_dir, shuffle=False):
    n_units = np.prod(n_units_per_layer[conv_idx])
    label_prototypes = [np.zeros(n_units) for l in range(n_labels)]
    label_count = [0 for l in range(n_labels)]

    for noise_type in os.listdir(activations_dir):
        if 'lvl' not in noise_type: continue
        activ_dir = f'{activations_dir}{noise_type}/'
        for results_file in os.listdir(activ_dir):
            if 'pt' not in results_file: continue
            results_filepath = f'{activ_dir}{results_file}'
            with h5py.File(results_filepath, 'r') as results:
                activ = np.array(results[f'{conv_idx}_{t}_activations'])
                labels = np.array(results['labels'])

            if shuffle:
                np.random.seed(shuffle_seed)
                np.random.shuffle(labels)

            n_data = labels.size 
            activ = activ.reshape((n_data, -1))
            for l, a in zip(labels, activ):
                l = int(l)
                label_prototypes[l] += a
                label_count[l] += 1
            del activ
            gc.collect()

    if shuffle: # Undo seed setting for label shuffles
        np.random.seed()

    # Calculate prototypes
    labels_to_use = []
    for l, count in enumerate(label_count):
        label_prototypes[l] /= count
        if count > 0:
            labels_to_use.append(l)
        else:
            print(f'Warning: label {l} contains no samples and will be skipped.')

    # Calculate and save label prototypes
    label_prototypes = np.array(label_prototypes)
    label_prototypes = label_prototypes[labels_to_use, :]
    prototype_results = {
        'labels': labels_to_use, 'label_count': label_count,
        'prototypes': label_prototypes}
    prototypes_filename = f'prototypes_conv{conv_idx}_t{t}'
    if shuffle:
        prototypes_filename += f'_shuffle{shuffle_seed}'
    with open(f'{pca_activations_dir}{prototypes_filename}.p', 'wb') as f:
        pickle.dump(prototype_results, f, protocol=4)

    # Run PCA and save PCA models
    print('Running PCA on label prototypes with (samples, features) shape:')
    print(label_prototypes.shape)
    pca = PCA()
    pca.fit(label_prototypes)
    pca_filename = f'PCA_conv{conv_idx}_t{t}'
    if shuffle:
        pca_filename += f'_shuffle{shuffle_seed}'
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

    for conv_idx in [1, 2, 3, 4, 5]:
        for t in [0]:
            print(f'====== PROCESSING LAYER {conv_idx}, TIMESTEP {t} ======')
            get_data_and_fit_PCA(conv_idx, t, pca_activations_dir, shuffle=shuffle)
            gc.collect()

