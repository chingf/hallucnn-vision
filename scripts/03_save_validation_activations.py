import sys
sys.path.insert(0, '..')
import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import h5py
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from utils import AddGaussianNoise, AddSaltPepperNoise
from utils import MagShuffle, PhaseShuffle, AllShuffle
from presnet import PResNet18V3NSeparateHP
from torchvision.models.resnet import resnet18 as ResNet
from torchvision.models import ResNet18_Weights

TASK_NAME = str(sys.argv[1]) # pnet
CKPT_EPOCH = int(sys.argv[2]) # 99

# Global variables
engram_dir = '/mnt/smb/locker/abbott-locker/hcnn_vision_resnet/'
dataset_root = '/mnt/smb/locker/abbott-locker/hcnn_vision/imagenet/'
ckpt_root = f'{engram_dir}checkpoints/'
hps_root = f'{engram_dir}hyperparams/'
activations_root = f'{engram_dir}validation_activations/'

TRAIN_MEAN = [0.485, 0.456, 0.406]
TRAIN_STD  = [0.229, 0.224, 0.225]
WEIGHT_PATTERN_N = f'{ckpt_root}{TASK_NAME}/'
WEIGHT_PATTERN_N += f'pnet_pretrained_pc*_{CKPT_EPOCH:03d}.pth'

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
n_timesteps = 5

# Helper functions
def load_pnet(
    net, weight_pattern, build_graph, random_init,
    ff_multiplier, fb_multiplier, er_multiplier, device='cuda:0'
    ):

    # Initialize PNet
    pnet = PResNet18V3NSeparateHP(
        net, build_graph=build_graph, random_init=random_init,
        ff_multiplier=ff_multiplier, fb_multiplier=fb_multiplier,
        er_multiplier=er_multiplier, register_backbone_hooks=True)

    # Load pcoder weights
    for pc in range(pnet.number_of_pcoders):
        pc_dict = torch.load(
            weight_pattern.replace('*',f'{pc+1}'), map_location='cpu')
        pc_dict = pc_dict['pcoderweights']
        if 'C_sqrt' not in pc_dict:
            pc_dict['C_sqrt'] = torch.tensor(-1, dtype=torch.float)
        getattr(pnet, f'pcoder{pc+1}').load_state_dict(pc_dict)

    # Set initial hyperparameters
    hyperparams = []
    for i in range(1, 6):
        hps = {}
        hps['ffm'] = ff_multiplier
        hps['fbm'] = fb_multiplier
        hps['erm'] = er_multiplier
        hyperparams.append(hps)
    pnet.set_hyperparameters(hyperparams)
    pnet.eval()
    pnet.to(device)
    return pnet

def get_best_pfile(hps_dir):
    best_pfile = None
    best_perf = -np.inf
    for pfile in os.listdir(hps_dir):
        if not pfile.endswith('.p'): continue
        with open(f'{hps_dir}{pfile}', 'rb') as f:
            results = pickle.load(f)
        for log_idx in range(len(results)-1, 0, -1):
            log = results[log_idx]
            if ('NoisyPerf' in log[0]) and (log[2]==n_timesteps):
                break
        mean_perf = np.mean([
            log[1] for log in results[log_idx-5:log_idx+1]])
        if mean_perf > best_perf:
            best_pfile = pfile
            best_perf = mean_perf
        print(mean_perf)
    print(f'best is {best_perf}')
    return best_pfile

def get_hps_from_pfile(pfile_path):
    hps = []
    with open(pfile_path, 'rb') as f:
        results = pickle.load(f)
    for log_idx in range(len(results)-1, 0, -1):
        log = results[log_idx]
        if log[0] == 'Hyperparam/pcoder5_memory':
            log_idx = log_idx - 19 # Start of hps log
            break
    for layer in range(1,6):
        assert(results[log_idx][0] == f'Hyperparam/pcoder{layer}_feedforward')
        layer_hps = {
            'ffm': results[log_idx][1],
            'fbm': results[log_idx+1][1],
            'erm': results[log_idx+2][1]
            }
        hps.append(layer_hps)
        log_idx = log_idx + 4
    return hps

# Transform args
all_noises = ["gaussian_noise", "impulse_noise", "none"]
noise_gens = [
    [AddGaussianNoise(std=0.50),
     AddGaussianNoise(std=1.00),
     AddGaussianNoise(std=1.50)],
    [AddSaltPepperNoise(probability=0.05),
     AddSaltPepperNoise(probability=0.15),
     AddSaltPepperNoise(probability=0.3)],
    [None],
    ]

n_units_per_layer = {
    1: (64, 112, 112),
    2: (64, 56, 56),
    3: (128, 28, 28),
    4: (256, 14, 14),
    5: (512, 7, 7)
    }

for nt_idx, noise_type in enumerate(all_noises):
    for ng_idx, noise_gen in enumerate(noise_gens[nt_idx]):
        noise_name = f'{noise_type}_lvl_{ng_idx+1}'
        hps_dir = f'{hps_root}{TASK_NAME}/{noise_name}/'
        
        # Get hps of best-performing iteration
        pfile = get_best_pfile(hps_dir)
        hps = get_hps_from_pfile(f'{hps_dir}{pfile}')
        
        # Set network
        net = ResNet(weights=ResNet18_Weights.IMAGENET1K_V1)
        pnet = load_pnet(
            net, WEIGHT_PATTERN_N,
            build_graph=True, random_init=False, ff_multiplier=0.33,
            fb_multiplier=0.33, er_multiplier=0.)
        pnet.set_hyperparameters(hps)
            
        # Set up transforms
        transform_seq = [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),]
        if noise_gen is not None:
            transform_seq.append(noise_gen)
        transform_seq = transforms.Compose(transform_seq)
        
        # If activations already calculated, skip
        activations_dir = f'{activations_root}{TASK_NAME}/{noise_name}/'
        os.makedirs(activations_dir, exist_ok=True)
        hdf5_path = f'{activations_dir}{pfile[:-2]}.hdf5'
        if os.path.exists(hdf5_path): continue
        
        # Load dataset
        np.random.seed(1)
        val_subset_indices = np.random.choice(50000, size=800, replace=False)
        np.random.seed()
        val_ds = ImageNet(dataset_root, split='val', transform=transform_seq)
        val_subset = torch.utils.data.Subset(val_ds, val_subset_indices)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1, drop_last=False)
        del val_ds
        print('ImageNet Loaded.')
        
        # Run and save
        with h5py.File(hdf5_path, 'x') as f_out:
            # Initialize hdf5 containers
            data_dict = {}
            for layer in np.arange(1, 6):
                activ_dim = (len(val_loader),) + n_units_per_layer[layer]
                for timestep in range(n_timesteps):
                    data_dict[f'{layer}_{timestep}_activations'] = f_out.create_dataset(
                        f'{layer}_{timestep}_activations', activ_dim, dtype='float32'
                        )
            data_dict['labels'] = f_out.create_dataset(
                'labels', len(val_loader), dtype='int')
            for timestep in range(5):
                data_dict[f'label_{timestep}'] = f_out.create_dataset(
                    f'label_{timestep}', len(val_loader), dtype='int')
                
            # Feed inputs into network
            for d_idx, (_in, _label) in enumerate(val_loader):
                pnet.reset()
                _in = _in.to(DEVICE)
                data_dict['labels'][d_idx] = _label.item()
                for t in range(n_timesteps): 
                    _in_t = _in if t == 0 else None
                    with torch.no_grad():
                        output = pnet(_in_t)
                    pred_label = output.max(-1)[1].item()
                    for layer in np.arange(1,6):
                        data_dict[f'{layer}_{t}_activations'][d_idx] = getattr(
                            pnet, f'block{layer}_repr').detach().cpu().numpy()
                    data_dict[f'label_{t}'][d_idx] = pred_label
