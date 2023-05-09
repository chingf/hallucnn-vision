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

########################
## ARGS 
########################
job_idx = int(sys.argv[1]) # 0-3
TASK_NAME = str(sys.argv[2]) # pnet
CKPT_EPOCH = 49

########################
## GPU management 
########################
batch_size = 32
try:
    n_gpus = (len(os.environ['CUDA_VISIBLE_DEVICES'])+1)/2
except:
    n_gpus = 0
#if n_gpus > 1:
#    device_num = str(job_idx % n_gpus)
#    my_env = os.environ
#    my_env["CUDA_VISIBLE_DEVICES"] = device_num
my_env = os.environ
my_env["CUDA_VISIBLE_DEVICES"] = str(job_idx)

########################
## GLOBAL CONFIGURATIONS
########################
if user == 'jwl':
    dataset_root = '/mnt/smb/locker/issa-locker/imagenet/'
    ckpt_root = '../hcnn-vision-files/checkpoints/'
else:
    dataset_root = '/mnt/smb/locker/issa-locker/imagenet/'
    ckpt_root = '../hcnn-vision-files/checkpoints/'

TRAIN_MEAN = [0.485, 0.456, 0.406]
TRAIN_STD  = [0.229, 0.224, 0.225]
if CKPT_EPOCH >= 100:
    ckpt_str = CKPT_EPOCH
else:
    ckpt_str = f'0{CKPT_EPOCH}'
WEIGHT_PATTERN_N = f'{ckpt_root}{TASK_NAME}/pnet_pretrained_pc*_{ckpt_str}.pth'
LOG_DIR = '../hps/'
LR_SCALE = 0.1
os.makedirs(f'{LOG_DIR}', exist_ok=True)

#total training epoches
EPOCH = 45
MAX_TIMESTEP = 5

#time of we run the script
TIME_NOW = datetime.now().isoformat()

########################
########################

def evaluate(net, epoch, dataloader, timesteps, writer=None, picklewriter=None, tag='Clean'):
    test_loss = np.zeros((timesteps+1,))
    correct   = np.zeros((timesteps+1,))
    for (images, labels) in dataloader:
        images = images.cuda()
        labels = labels.cuda()
        
        with torch.no_grad():
            for tt in range(timesteps+1):
                if tt == 0:
                    outputs = net(images)
                else:
                    outputs = net()
                loss = loss_function(outputs, labels)
                test_loss[tt] += loss.item()
                _, preds = outputs.max(1)
                correct[tt] += preds.eq(labels).sum()

    print()
    for tt in range(timesteps+1):
        test_loss[tt] /= len(dataloader.dataset)
        correct[tt] /= len(dataloader.dataset)
        print('Test set t = {:02d}: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            tt,
            test_loss[tt],
            correct[tt]
        ))
        if writer is not None:
            writer.add_scalar(f"{tag}Perf/Epoch#{epoch}", correct[tt], tt)
        if picklewriter is not None:
            picklewriter.append([f"{tag}Perf/Epoch#{epoch}", correct[tt], tt])
    print()

def train(net, epoch, dataloader, timesteps, writer=None, picklewriter=None):
    for batch_index, (images, labels) in enumerate(dataloader):
        net.reset()

        labels = labels.cuda()
        images = images.cuda()

        ttloss = np.zeros((timesteps+1))
        optimizer.zero_grad()

        for tt in range(timesteps+1):
            if tt == 0:
                outputs = net(images)
                loss = loss_function(outputs, labels)
                ttloss[tt] = loss.item()
            else:
                outputs = net()
                current_loss = loss_function(outputs, labels)
                ttloss[tt] = current_loss.item()
                loss += current_loss
        
        loss.backward()
        optimizer.step()
        net.update_hyperparameters()
            
        print(f"Training Epoch: {epoch} [{batch_index * batch_size + len(images)}/{len(dataloader.dataset)}]\tLoss: {loss.item():0.4f}\tLR: {optimizer.param_groups[0]['lr']:0.6f}")
        for tt in range(timesteps+1):
            print(f'{ttloss[tt]:0.4f}\t', end='')
        print()
        if writer is not None:
            writer.add_scalar(f"TrainingLoss/CE", loss.item(), (epoch-1)*len(dataloader) + batch_index)
        if picklewriter is not None:
            picklewriter.append([f"TrainingLoss/CE", loss.item(), (epoch-1)*len(dataloader) + batch_index])

def load_pnet(
    net, weight_pattern, build_graph, random_init,
    ff_multiplier, fb_multiplier, er_multiplier, same_param, device='cuda:0'
    ):

    if same_param:
        raise Exception('Not implemented!')
    else:
        pnet = PEffN_b0SeparateHP_V1(
            net, build_graph=build_graph, random_init=random_init,
            ff_multiplier=ff_multiplier, fb_multiplier=fb_multiplier, er_multiplier=er_multiplier)

    for pc in range(pnet.number_of_pcoders):
        pc_dict = torch.load(weight_pattern.replace('*',f'{pc+1}'), map_location='cpu')
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

args = []
for nt_idx, noise_type in enumerate(all_noises):
    for ng_idx, noise_gen in enumerate(noise_gens[nt_idx]):
        args.append([nt_idx, noise_type, ng_idx, noise_gen])
split_args = np.array_split(args, n_gpus)
job_args = split_args[job_idx]
for job_arg in job_args:
    nt_idx, noise_type, ng_idx, noise_gen = job_arg
    print(noise_gen)
    start = datetime.now()
    
    noise_level = 0
    transform_clean = [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    transform_noise = transform_clean[:]

    transform_clean.append(transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD))
    transform_noise.append(transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD))

    if noise_gen is not None:
        noise_level = ng_idx + 1
        transform_noise.append(noise_gen)

    np.random.seed(0)
    subset_indices = np.random.choice(1281167, size=30000, replace=False)
    np.random.seed()
    clean_ds = ImageNet(dataset_root, split='train', transform=transforms.Compose(transform_clean))
    clean_subset = torch.utils.data.Subset(clean_ds, subset_indices)
    clean_loader = torch.utils.data.DataLoader(
        clean_subset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    noise_ds = ImageNet(dataset_root, split='train', transform=transforms.Compose(transform_noise))
    noise_subset = torch.utils.data.Subset(noise_ds, subset_indices)
    noise_loader = torch.utils.data.DataLoader(
        noise_subset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)

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
    
    backward_weight_patter = WEIGHT_PATTERN_N

    # feedforward for baseline
    print(datetime.now() - start)
    loss_function = nn.CrossEntropyLoss()
    net = efficientnet_b0(pretrained=True)
    pnet = load_pnet(net, backward_weight_patter,
        build_graph=True, random_init=False, ff_multiplier=0.33,
        fb_multiplier=0.33, er_multiplier=0.0, same_param=False, device='cuda:0')

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

    log_hyper_parameters(pnet, 0, sumwriter, picklewriter)
    hps = pnet.get_hyperparameters_values()
    print(hps)

    evaluate(
        pnet, 0, noise_loader, timesteps=MAX_TIMESTEP,
        writer=sumwriter, picklewriter=picklewriter, tag='Noisy')
    print(datetime.now() - start)
    for epoch in range(1, EPOCH+1):
        train(
            pnet, epoch, noise_loader, timesteps=MAX_TIMESTEP,
            writer=sumwriter, picklewriter=picklewriter)
        print(datetime.now() - start)
        log_hyper_parameters(pnet, epoch, sumwriter, picklewriter)

        hps = pnet.get_hyperparameters_values()
        print(hps)

        evaluate(
            pnet, epoch, noise_loader, timesteps=MAX_TIMESTEP,
            writer=sumwriter, picklewriter=picklewriter, tag='Noisy')
        if picklewriter is not None:
            with open(picklewriter_file, 'wb') as f:
                pickle.dump(picklewriter, f)
        print(datetime.now() - start)

    evaluate(
        pnet, epoch, clean_loader, timesteps=MAX_TIMESTEP,
        writer=sumwriter, picklewriter=picklewriter, tag='Clean')
    
    if sumwriter is not None:
        sumwriter.close()

    if picklewriter is not None:
        with open(picklewriter_file, 'wb') as f:
            pickle.dump(picklewriter, f)

    del pnet
    gc.collect()
    print(datetime.now() - start)

