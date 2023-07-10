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
split = str(sys.argv[2]) # 0, there are 10 total
if len(sys.argv) > 3:
    shuffle_seed = int(sys.argv[3])
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
n_images = 40000
train_data_splits = np.split(np.arange(40), 10)

def get_data_and_fit_PCA(conv_idx, t, pca_activations_dir, shuffle=False):
    n_units = np.prod(n_units_per_layer[conv_idx])
    image_prototypes = np.zeros((n_images, n_units), dtype=np.float32)
    image_count = np.zeros(n_images, dtype=np.int16)
    chosen_splits = train_data_splits[split]
    for noise_type in os.listdir(activations_dir):
        if 'lvl' not in noise_type: continue
        activ_dir = f'{activations_dir}{noise_type}/'
        idx_offset = 0
        for results_file in os.listdir(activ_dir):
            choose_file = False
            for chosen_split in chosen_splits:
                if results_file.endswith(f'pt{chosen_split}.hdf5'):
                    choose_file = True
                    break
            if not choose_file: continue
            print(f'Processing: {noise_type}, {results_file}')
            results_filepath = f'{activ_dir}{results_file}'
            with h5py.File(results_filepath, 'r') as results:
                activ = np.array(results[f'{conv_idx}_{t}_activations'])
                labels = np.array(results['labels'])
            n_data = labels.size 
            activ = activ.reshape((n_data, -1))

            if shuffle:
                np.random.seed(shuffle_seed)
                shuffle_indices = np.arange(n_data)
                np.random.shuffle(shuffle_indices)
                activ = activ[shuffle_indices]

            for i, a in enumerate(activ):
                image_prototypes[i+idx_offset] += a
                image_count[i+idx_offset] += 1
            idx_offset += 1000
            del activ
            del labels
            del a
            gc.collect()

    if shuffle: # Undo seed setting for image shuffles
        np.random.seed()

    # Calculate prototypes
    images_to_use = []
    for i, count in enumerate(image_count):
        image_prototypes[i] /= count
        if count > 0:
            images_to_use.append(i)
        else:
            print(f'Warning: image {i} contains no samples and will be skipped.')

    # Calculate and save image prototypes
    image_prototypes = image_prototypes[images_to_use, :]
    prototype_results = {
        'images': images_to_use, 'image_count': image_count,
        'prototypes': image_prototypes}
    prototypes_filename = f'image_prototypes_conv{conv_idx}_t{t}_split{split}'
    if shuffle:
        prototypes_filename += f'_shuffle{shuffle_seed}'
    with open(f'{pca_activations_dir}{prototypes_filename}.p', 'wb') as f:
        pickle.dump(prototype_results, f, protocol=4)

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

