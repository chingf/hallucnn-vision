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

# Arguments
netname = str(sys.argv[1]) # pnet
split = str(sys.argv[2]) # 0, there are 10 total
if len(sys.argv) > 3:
    shuffle_seed = int(sys.argv[3])
    print(f'Shuffle-label PCA with seed {shuffle_seed}')
    shuffle = True
else:
    shuffle = False
auc = True
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn_vision_resnet/'
dataset_root = '/mnt/smb/locker/abbott-locker/hcnn_vision/imagenet/'
ckpt_root = f'{engram_dir}checkpoints/'
hps_root = f'{engram_dir}hyperparams/'
train_activations_dir = f'{engram_dir}train_activations/{netname}/'
validation_activations_dir = f'{engram_dir}activations/{netname}/'
pca_activations_dir = f'{engram_dir}train_prototype_PCA/{netname}/'
pickles_dir = f'{engram_dir}pickles/'

# Helper functions
train_data_splits = np.split(np.arange(40), 10)
def get_cpu_usage():
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t --giga').readlines()[-1].split()[1:])

    # Memory usage
    p_used = round((used_memory/total_memory) * 100, 2)
    print(f"RAM {used_memory} GB, {p_used}% used")

def subsample_train_data(conv_idx, t, noise_string):
    activ_dir = f'{validation_activations_dir}{noise_string}/'
    chosen_splits = train_data_splits[split]
    for results_file in os.listdir(activ_dir):
        choose_file = False
        for chosen_split in chosen_splits:
            if results_file.endswith(f'pt{chosen_split}.hdf5'):
                choose_file = True
                break
        if not choose_file: continue
        results_filepath = f'{activ_dir}{results_file}'
        results = h5py.File(results_filepath, 'r')
    activ = np.array(results[f'{conv_idx}_{t}_activations'])
    n_data = activ.shape[0]
    activ = activ.reshape((n_data, -1))
    return activ, np.array(results['labels'])

def get_projection(activ, pca):
    activ_centered = activ - pca.mean_[None,:]
    projected_activ = activ_centered @ (pca.components_).T
    return projected_activ

def get_explained_var(centered_activ, pca, auc=True):
    """ ACTIV should be of shape (N, DIMS)"""
    
    sample_size = centered_activ.shape[0]
    projected_activ = centered_activ @ pca.components_.T
    total_var = np.sum(np.mean(np.square(projected_activ), axis=0))
    var_by_component = np.mean(np.square(projected_activ), axis=0)/total_var
    if auc:
        var_curve = np.cumsum(var_by_component)
        explained_var = np.trapz(var_curve, dx=1/var_curve.size)
    else:
        pca_cum_var = np.cumsum(pca.explained_variance_ratio_)
        K = np.argwhere(pca_cum_var>0.9)[0].item()
        explained_var = np.sum(var_by_component[:K+1])
    return explained_var 

def main():
    # Measure factorization for each noise/layer/timestep
    convs = []
    ts = []
    factorization = []

    for conv_idx in [1,2,3,4,5]:
        # Load PCA model and the prototype vectors from t = 0
        prototypes_fname = f'prototypes_conv{conv_idx}_t0'
        if shuffle:
            prototypes_fname += f'_shuffle{shuffle_seed}'
        prototypes_fname = f'{pca_activations_dir}{prototypes_fname}.p'
        with open(prototypes_fname, 'rb') as f:
            prototype_results = pickle.load(f)
        labels_to_use = prototype_results['labels']
        prototypes = prototype_results['prototypes']
        pca_fname = f'PCA_conv{conv_idx}_t0'
        if shuffle:
            pca_fname += f'_shuffle{shuffle_seed}' 
        with open(f'{pca_activations_dir}{pca_fname}.p', 'rb') as f:
            pca = pickle.load(f)

        # Iterate over timesteps of predictive processing
        for t in [0,1,2,3,4]:
            activ = []
            label = []
            for noise_string in os.listdir(validation_activations_dir):
                if 'lvl' not in noise_string: continue
                print(f'Conv {conv_idx}, t {t}, {noise_string}')
                _activ, _label = subsample_train_data(conv_idx, t, noise_string)
                activ.append(_activ)
                label.append(_label)
            del _activ
            del _label
            gc.collect()
            activ = np.vstack(activ)
            label = np.concatenate(label)
            if shuffle:
                np.random.seed(shuffle_seed+1)
                np.random.shuffle(label)
            print(f'Activations shape: {activ.shape}, for {label.shape} labels')

            # Calculate factorization for each centered label
            print(f'There are {labels_to_use} labels used.')
            var_ratios = []
            for l_idx, l in enumerate(labels_to_use):
                activ_indices = label==l
                if np.sum(activ_indices) == 0:
                    continue
                prototype = prototypes[l_idx]
                centered_activ = activ[activ_indices] - prototype[None,:]
                l_var_ratio = get_explained_var(centered_activ, pca, auc=auc)
                if not auc:
                    l_var_ratio = l_var_ratio / 0.9
                convs.append(conv_idx)
                ts.append(t)
                factorization.append(l_var_ratio)
        
            del activ
            del centered_activ
            del label
            gc.collect()
            get_cpu_usage()
            
    df = pd.DataFrame({
        'Conv': convs,
        'T': ts,
        'Factorization': factorization
        })
    os.makedirs(pickles_dir, exist_ok=True)
    pfile = f'factorization_split{split}.p'
    if auc:
        pfile = 'auc_' + pfile
    if shuffle:
        pfile = 'shuffle_' + pfile
    pfile = f'{pickles_dir}{netname}_{pfile}'
    with open(pfile, 'wb') as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    main()

