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
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn_vision/'
activations_dir = f'{engram_dir}activations/{netname}/'
pickles_dir = f'{engram_dir}pickles/'
bg_types = ['gaussian_noise', 'impulse_noise']
snr_types = [1, 2, 3]

def calc_inv():
    bgs = []
    snrs = []
    convs = []
    ts = []
    clean_noise_dist = []
    clean_clean_dist = []

    clean_activ_dir = f'{activations_dir}none_lvl_0/'
    for results_file in os.listdir(clean_activ_dir):
        results_filepath = f'{clean_activ_dir}{results_file}'
        clean_results = h5py.File(results_filepath, 'r')

    for conv_idx in [1,2,3,4,5,6,7,8]:
        for t in [0,1,2,3,4]:
            clean_activ = np.array(clean_results[f'{conv_idx}_{t}_activations'])
            n_data = clean_activ.shape[0]
            clean_activ = clean_activ.reshape((n_data, -1))

            for bg in bg_types:
                for snr in snr_types:
                    activ_dir = f'{activations_dir}{bg}_lvl_{snr}/'
                    for results_file in os.listdir(activ_dir):
                        results_filepath = f'{activ_dir}{results_file}'
                        results = h5py.File(results_filepath, 'r')
                    activ = np.array(results[f'{conv_idx}_{t}_activations'])
                    n_data = activ.shape[0]
                    activ = activ.reshape((n_data, -1))

                    bgs.append(bg)
                    snrs.append(snr)
                    convs.append(conv_idx)
                    ts.append(t)
                   
                    # Clean-noisy distance for same image
                    _clean_noise_dist = []
                    for idx in range(len(activ)):
                        _clean_noise_dist.append(
                            np.square(activ[idx] - clean_activ[idx]))
                    _clean_noise_dist = np.mean(_clean_noise_dist, axis=0)
                    _clean_noise_dist = np.sum(_clean_noise_dist)
                    clean_noise_dist.append(_clean_noise_dist)
                    
                    # Distance between any two clean images
                    _clean_clean_dist = []
                    for idx in range(len(activ)):
                        rand_idx = np.random.choice(len(activ))
                        _clean_clean_dist.append(np.square(
                            clean_activ[idx] - clean_activ[rand_idx]))
                    _clean_clean_dist = np.mean(_clean_clean_dist, axis=0)
                    _clean_clean_dist = np.sum(_clean_clean_dist)
                    clean_clean_dist.append(_clean_clean_dist)
    
    df = pd.DataFrame({
        'BG': bgs,
        'SNR': snrs,
        'Conv': convs,
        'T': ts,
        'Dist': clean_noise_dist,
        'Invariance by Radius': np.array(clean_noise_dist)/np.array(clean_clean_dist),
        })

    os.makedirs(pickles_dir, exist_ok=True)
    pfile = f'{pickles_dir}{netname}_invariance.p'
    with open(pfile, 'wb') as f:
        pickle.dump(df, f)

calc_inv()
