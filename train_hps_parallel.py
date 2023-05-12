import sys
import os
import torch
import torchvision.transforms as transforms
from   torchvision.datasets import ImageNet
from   datetime import datetime
import torch.optim as optim
import torch.nn as nn
import gc
import shortuuid
import pickle
from contextlib import closing
import torch.multiprocessing as mp
import torch.distributed as dist
import socket
import gc

import numpy as np
from   utils import AddGaussianNoise, AddSaltPepperNoise
from   timm.models import efficientnet_b0
from   peff_b0 import PEffN_b0SeparateHP_V1
import tensorboard
if os.environ['USER'] == 'jwl2182':
    user = 'jwl'
else:
    user = 'cf'

if user == 'cf':
    from torch.utils.tensorboard import SummaryWriter

def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # Initialize the process group.
    dist.init_process_group('NCCL', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


########################
## ARGS 
########################
TASK_NAME = str(sys.argv[1]) # pnet
CKPT_EPOCH = int(sys.argv[2]) # 49

########################
## GLOBAL CONFIGURATIONS
########################
batch_size = 11 # Effect size 32
try:
    n_gpus = int((len(os.environ['CUDA_VISIBLE_DEVICES'])+1)/2)
except:
    raise ValueError('GPUs needed')
if user == 'jwl':
    dataset_root = '/mnt/smb/locker/issa-locker/imagenet/'
    ckpt_root = '../hcnn-vision-files/checkpoints/'
    LOG_DIR = '../hps/'
else:
    engram_dir = '/mnt/smb/locker/abbott-locker/hcnn_vision/'
    dataset_root = f'{engram_dir}imagenet/'
    ckpt_root = f'{engram_dir}checkpoints/'
    LOG_DIR = f'{engram_dir}hyperparams/'
os.makedirs(LOG_DIR, exist_ok=True)
TRAIN_MEAN = [0.485, 0.456, 0.406]
TRAIN_STD  = [0.229, 0.224, 0.225]
WEIGHT_PATTERN_N = f'{ckpt_root}{TASK_NAME}/'
WEIGHT_PATTERN_N += f'pnet_pretrained_pc*_{CKPT_EPOCH:03d}.pth'
LR_SCALE = 0.1
EPOCH = 45 # Total training epochs
MAX_TIMESTEP = 5

########################
########################

def evaluate(
    ddpnet, net, epoch, dataloader, timesteps,
    loss_function, writer=None, picklewriter=None, tag='Clean'):

    test_loss = np.zeros((timesteps+1,))
    correct   = np.zeros((timesteps+1,))
    for (images, labels) in dataloader:
        net.reset()
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        with torch.no_grad():
            for tt in range(timesteps+1):
                if tt == 0:
                    outputs = ddpnet(images)
                else:
                    outputs = ddpnet(None)
                loss = loss_function(outputs, labels)
                test_loss[tt] += loss.item()
                _, preds = outputs.max(1)
                correct[tt] += preds.eq(labels).sum()
    print()
    for tt in range(timesteps+1):
        test_loss[tt] /= (len(dataloader.dataset)/n_gpus)
        correct[tt] /= (len(dataloader.dataset)/n_gpus)
        print('Test set t = {:02d}: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            tt, test_loss[tt], correct[tt]))
        if writer is not None:
            writer.add_scalar(f"{tag}Perf/Epoch#{epoch}", correct[tt], tt)
        if picklewriter is not None:
            picklewriter.append([f"{tag}Perf/Epoch#{epoch}", correct[tt], tt])
    print()

def train(
    ddpnet, net, epoch, dataloader, timesteps, loss_function, optimizer,
    gpu, writer=None, picklewriter=None):

    for batch_index, (images, labels) in enumerate(dataloader):
        net.reset()

        labels = labels.cuda()
        images = images.cuda()

        ttloss = np.zeros((timesteps+1))
        optimizer.zero_grad()

        for tt in range(timesteps+1):
            if tt == 0:
                outputs = ddpnet(images)
                loss = loss_function(outputs, labels)
                ttloss[tt] = loss.item()
            else:
                outputs = ddpnet(None)
                current_loss = loss_function(outputs, labels)
                ttloss[tt] = current_loss.item()
                loss = loss + current_loss
        loss.backward()
        optimizer.step()
        net.update_hyperparameters()

        dset_len = len(dataloader.dataset)/n_gpus
        print_string = f'Training Epoch: {epoch} '
        print_string += f'[{batch_index * batch_size + len(images)}/{dset_len}]'
        print_string += f'\tLoss: {loss.item():0.4f}'
        print(print_string)
        for tt in range(timesteps+1):
            print(f'{ttloss[tt]:0.4f}\t', end='')
        print()
        if writer is not None:
            writer.add_scalar(
                f"TrainingLoss/CE", loss.item(), (epoch-1)*dset_len + batch_index)
        if picklewriter is not None:
            picklewriter.append([
                f"TrainingLoss/CE", loss.item(), (epoch-1)*dset_len + batch_index])

def load_pnet(
    net, weight_pattern, build_graph, random_init,
    ff_multiplier, fb_multiplier, er_multiplier, same_param, device='cuda:0'
    ):

    if same_param:
        raise Exception('Not implemented!')
    else:
        pnet = PEffN_b0SeparateHP_V1(
            net, build_graph=build_graph, random_init=random_init,
            ff_multiplier=ff_multiplier, fb_multiplier=fb_multiplier,
            er_multiplier=er_multiplier)

    for pc in range(pnet.number_of_pcoders):
        pc_dict = torch.load(
            weight_pattern.replace('*',f'{pc+1}'), map_location='cpu')
        pc_dict = pc_dict['pcoderweights']
        if 'C_sqrt' not in pc_dict:
            pc_dict['C_sqrt'] = torch.tensor(-1, dtype=torch.float)
        getattr(pnet, f'pcoder{pc+1}').load_state_dict(pc_dict)

    pnet.eval()
    pnet.to(device)
    return pnet

def log_hyper_parameters(net, epoch, sumwriter=None, picklewriter=None):
    if sumwriter is not None:
        for i in range(1, net.number_of_pcoders+1):
            sumwriter.add_scalar(
                f"Hyperparam/pcoder{i}_feedforward", getattr(net,f'ffm{i}').item(), epoch)
            if i < net.number_of_pcoders:
                sumwriter.add_scalar(
                    f"Hyperparam/pcoder{i}_feedback", getattr(net,f'fbm{i}').item(), epoch)
            else:
                sumwriter.add_scalar(
                    f"Hyperparam/pcoder{i}_feedback", 0, epoch)
            sumwriter.add_scalar(
                f"Hyperparam/pcoder{i}_error", getattr(net,f'erm{i}').item(), epoch)
            if i < net.number_of_pcoders:
                sumwriter.add_scalar(
                    f"Hyperparam/pcoder{i}_memory",
                    1-getattr(net,f'ffm{i}').item()-getattr(net,f'fbm{i}').item(), epoch)
            else:
                sumwriter.add_scalar(
                    f"Hyperparam/pcoder{i}_memory", 1-getattr(net,f'ffm{i}').item(), epoch)
    if picklewriter is not None:
        for i in range(1, net.number_of_pcoders+1):
            picklewriter.append([
                f"Hyperparam/pcoder{i}_feedforward", getattr(net,f'ffm{i}').item(), epoch])
            if i < net.number_of_pcoders:
                picklewriter.append([
                    f"Hyperparam/pcoder{i}_feedback", getattr(net,f'fbm{i}').item(), epoch])
            else:
                picklewriter.append([f"Hyperparam/pcoder{i}_feedback", 0, epoch])
            picklewriter.append([
                f"Hyperparam/pcoder{i}_error", getattr(net,f'erm{i}').item(), epoch])
            if i < net.number_of_pcoders:
                picklewriter.append([
                    f"Hyperparam/pcoder{i}_memory",
                    1-getattr(net,f'ffm{i}').item()-getattr(net,f'fbm{i}').item(), epoch])
            else:
                picklewriter.append([
                    f"Hyperparam/pcoder{i}_memory", 1-getattr(net,f'ffm{i}').item(), epoch])

def train_and_eval(gpu, mp_args):
    # Unpack args
    noise_type = mp_args['noise_type']
    noise_gen = mp_args['noise_gen']
    noise_level = mp_args['noise_level']
    free_port = mp_args['free_port']
    cuda_device = torch.device('cuda', gpu)

    # GPU Set up
    setup(gpu, n_gpus, free_port)
    torch.cuda.set_device(gpu)

    # Load network
    net = efficientnet_b0(pretrained=True)
    pnet = load_pnet( # TODO: random hyperparam initialization
        net, WEIGHT_PATTERN_N,
        build_graph=True, random_init=False, ff_multiplier=0.33,
        fb_multiplier=0.33, er_multiplier=0.0, same_param=False)
    pnet.cuda()

    # Distributed data parallel network
    ddpnet = torch.nn.parallel.DistributedDataParallel(
        pnet, device_ids=[gpu], broadcast_buffers=False,
        find_unused_parameters=True)

    # Set up optimizer
    loss_function = nn.CrossEntropyLoss()
    hyperparams = [*pnet.get_hyperparameters()]
    fffbmem_hp = []
    erm_hp = []
    for pc in range(pnet.number_of_pcoders):
        fffbmem_hp.extend(hyperparams[pc*4:pc*4+3])
        erm_hp.append(hyperparams[pc*4+3])
    optimizer = optim.Adam([
        {'params': fffbmem_hp, 'lr':0.01*LR_SCALE},
        {'params': erm_hp, 'lr':0.0001*LR_SCALE}], weight_decay=0.00001)

    # Set up transforms
    transform_clean = [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),]
    transform_noise = transform_clean[:]
    if noise_gen is not None:
        transform_noise.append(noise_gen)

    # Set up data
    np.random.seed(0)
    subset_indices = np.random.choice(1281167, size=30000, replace=False)
    np.random.seed()
    print('Loading ImageNet')
    noise_ds = ImageNet(
        dataset_root, split='train',
        transform=transforms.Compose(transform_noise))
    noise_subset = torch.utils.data.Subset(noise_ds, subset_indices)
    del noise_ds
    gc.collect()
    print('ImageNet Loaded.')
    noise_sampler = torch.utils.data.distributed.DistributedSampler(
        noise_subset, num_replicas=n_gpus, rank=gpu)
    noise_loader = torch.utils.data.DataLoader(
        noise_subset, batch_size=batch_size,
        drop_last=False, num_workers=4, sampler=noise_sampler)

    # Initial logging set up and step
    if gpu == 0:
        # TODO: set up proper validation dataset
        print('STILL NEED TO SET UP PROPER VALID SET')

        if False: #user == 'cf': # For now, don't use SummaryWriter
            sumwriter = SummaryWriter(
                    f'{LOG_DIR}{TASK_NAME}_type_{noise_type}_lvl_{noise_level}',
                    filename_suffix=f'_{noise_type}_{noise_level}')
            picklewriter = None
        else:
            sumwriter = None
            textwriter_dir = f'{LOG_DIR}{TASK_NAME}/{noise_type}_lvl_{noise_level}/'
            os.makedirs(textwriter_dir, exist_ok=True)
            unique_id = shortuuid.uuid()
            picklewriter = []
            picklewriter_file = f'{textwriter_dir}{LR_SCALE}x_{unique_id}.p'

        log_hyper_parameters(pnet, 0, sumwriter, picklewriter)
        hps = pnet.get_hyperparameters_values()
        print(hps)
        evaluate(
            ddpnet, pnet, 0, noise_loader, MAX_TIMESTEP,
            loss_function, sumwriter, picklewriter, tag='Noisy')
    else:
        sumwriter = picklewriter = None

    # Train/eval loop
    start = datetime.now()
    for epoch in range(1, EPOCH+1):
        if gpu == 0:
            print(f'====== TRAIN EPOCH {epoch} =====')
        train(
            ddpnet, pnet, epoch, noise_loader, MAX_TIMESTEP,
            loss_function, optimizer, gpu, sumwriter, picklewriter)
        if gpu == 0:
            print(datetime.now() - start)
            print(f'====== EVAL EPOCH {epoch} =====')
            log_hyper_parameters(pnet, epoch, sumwriter, picklewriter)
            hps = pnet.get_hyperparameters_values()
            print('Hyperparameters:')
            print(hps)
            evaluate(
                ddpnet, pnet, epoch, noise_loader, MAX_TIMESTEP,
                loss_function, sumwriter, picklewriter, tag='Noisy')
            if picklewriter is not None:
                with open(picklewriter_file, 'wb') as f:
                    pickle.dump(picklewriter, f)
            print(datetime.now() - start)

    cleanup()
    if gpu == 0 and sumwriter != None:
        sumwriter.close()

#####################################################
#              BEGIN SCRIPT
#####################################################
if __name__ == '__main__':
    free_port = get_open_port()
    all_noises = ["gaussian_noise", "impulse_noise", "none"]
    noise_gens = [
        [
            AddGaussianNoise(std=0.50),
            AddGaussianNoise(std=1.00),
            AddGaussianNoise(std=1.50),
        ],
        [
            AddSaltPepperNoise(probability=0.05),
            AddSaltPepperNoise(probability=0.15),
            AddSaltPepperNoise(probability=0.3),
        ],
        [None],
    ]

    # For each noise combo, run parallelized hyperparameter training
    for nt_idx, noise_type in enumerate(all_noises):
        for ng_idx, noise_gen in enumerate(noise_gens[nt_idx]):
            mp_args = {}
            mp_args['noise_type'] = noise_type
            mp_args['noise_gen'] = noise_gen
            mp_args['noise_level'] = ng_idx + 1
            mp_args['free_port'] = free_port
            mp.spawn(train_and_eval, nprocs=n_gpus, args=(mp_args,))

