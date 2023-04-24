#########################
# In this script we train p-EfficientNets on ImageNet
# We use the pretrained model and only train feedback connections.
# This uses data parallelization across multiple GPUs.
#########################
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import os
import time
from contextlib import closing
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.multiprocessing as mp
import torch.distributed as dist
import socket
import gc

from timm.models import efficientnet_b0
from utils import AddGaussianNoise, AddSaltPepperNoise
from peff_b0 import PEffN_b0SeparateHP_V1

################################################
#       Global configs
################################################

engram_dir = '/mnt/smb/locker/abbott-locker/hcnn_vision/'
NUMBER_OF_PCODERS = 8
transform_val = transforms.Compose([ # Dataset and train-test helpers
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ElasticTransform(alpha=250., sigma=4.)
])
data_root = '/mnt/smb/locker/abbott-locker/hcnn_vision/imagenet/'

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

class Args():

    def __init__(self):
        self.random_seed = None                    #random_seed for the run
        self.batchsize = 80                        #batchsize for training
        self.num_workers = 4                       #number of workers
        self.num_epochs = 50                       #number of epochs
        self.num_gpus = 3 
        self.start_epoch = 1

        self.task_name =  'pnet_elastic'       #dir_name
        self.extra_stuff_you_want_to_add_to_tb = ' '
        self.log_dir = f'{engram_dir}tensorboard/{self.task_name}/'       #tensorboard logdir
        self.pth_dir = f'{engram_dir}checkpoints/{self.task_name}/'       #ckpt dir

        self.optim_name = 'RMSProp'
        self.lr = 0.001 * (64/self.batchsize)
        self.weight_decay = 5e-4
        self.ckpt_every = None   #TODO

        # optional
        self.resume = None                         #resuming the training 
        #path to the checkpoints. Should be a list of len equal to NUMBER_OF_PCODERS
        self.resume_ckpts= [f"../weights/PEffNetB0/pnet_pretrained_pc{x+1}_001.pth" for x in range(8)]


def train_pcoders(
    net, epoch, loss_function, optimizer, writer, train_loader, args, verbose=True
    ):

    ''' A training epoch '''
    
    net.train()

    tstart = time.time()
    for batch_index, (images, _) in enumerate(train_loader):
        net.reset()
        images = images.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        for i in range(NUMBER_OF_PCODERS):
            if i == 0:
                a = loss_function(net.pcoder1.prd, images)
                loss = a
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_curr = getattr(net, f"pcoder{i+1}")
                a = loss_function(pcoder_curr.prd, pcoder_pre.rep)
                loss += a
            if writer != None:
                writer.add_scalar(
                    f"MSE Train/PCoder{i+1}", a.item(), epoch * len(train_loader) + batch_index)

        loss.backward()
        optimizer.step()

        if writer != None:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.batchsize + len(images),
                total_samples=len(train_loader.dataset)
            ))
            print ('Time taken:',time.time()-tstart)
            writer.add_scalar(f"MSE Train/Sum", loss.item(), epoch * len(train_loader) + batch_index)


def test_pcoders(net, epoch, loss_function, optimizer, writer, test_loader, args, verbose=True):

    ''' A testing epoch '''

    net.eval()

    tstart = time.time()
    final_loss = [0 for i in range(NUMBER_OF_PCODERS)]
    for batch_index, (images, _) in enumerate(test_loader):
        net.reset()
        images = images.cuda()
        with torch.no_grad():
            outputs = net(images)
        for i in range(NUMBER_OF_PCODERS):
            if i == 0:
                final_loss[i] += loss_function(net.pcoder1.prd, images).item()
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_curr = getattr(net, f"pcoder{i+1}")
                final_loss[i] += loss_function(pcoder_curr.prd, pcoder_pre.rep).item()
    
    loss_sum = 0
    for i in range(NUMBER_OF_PCODERS):
        final_loss[i] /= len(test_loader)
        loss_sum += final_loss[i]
        if writer != None:
            writer.add_scalar(f"MSE Test/PCoder{i+1}", final_loss[i], epoch * len(test_loader))
    if writer != None:
        writer.add_scalar(f"MSE Test/Sum", loss_sum, epoch * len(test_loader))
        print('Testing Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss_sum,
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batchsize + len(images),
            total_samples=len(test_loader.dataset)
        ))
        print ('Time taken:',time.time()-tstart)

def train_and_eval(gpu, mp_args):
    # Unpack args
    train_subset_indices = mp_args['train_subset_indices']
    val_subset_indices = mp_args['val_subset_indices']
    free_port = mp_args['free_port']
    args = mp_args['args']
    NUMBER_OF_PCODERS = mp_args['NUMBER_OF_PCODERS']
    cuda_device = torch.device('cuda', gpu)

    # GPU Set up
    setup(gpu, args.num_gpus, free_port)
    torch.cuda.set_device(gpu)

    # Load network
    net = efficientnet_b0(pretrained=True)
    print("Loaded EffB0")
    pnet = PEffN_b0SeparateHP_V1(net, build_graph=True, random_init=False)
    print("Loaded Predictive-EffB0")
    pnet.cuda()

    # Distributed data parallel
    ddp_pnet = torch.nn.parallel.DistributedDataParallel(
        pnet, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True)

    # Set up optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.RMSprop([{
        'params':getattr(pnet,f"pcoder{x+1}").pmodule.parameters()
        } for x in range(NUMBER_OF_PCODERS)],
        lr=args.lr, weight_decay=args.weight_decay)
    
    # Set up data
    print('Loading train ds')
    train_ds = ImageNet(data_root, split='train', transform=transform_val)
    train_subset = torch.utils.data.Subset(train_ds, train_subset_indices)
       
    print('Loading val ds')
    val_ds = ImageNet(data_root, split='val', transform=transform_val)
    val_subset = torch.utils.data.Subset(val_ds, val_subset_indices)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_subset, num_replicas=args.num_gpus, rank=gpu)
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=args.batchsize, num_workers=args.num_workers,
        pin_memory=True, sampler=train_sampler, drop_last=False)
    if gpu == 0:
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=args.batchsize, num_workers=args.num_workers,
            pin_memory=True, drop_last=False)

    # Set up tensorboard logging
    if gpu == 0:
        sumwriter = SummaryWriter(args.log_dir, filename_suffix=f'')
        optimizer_text =  f"Optimizer   :{args.optim_name}  \n"
        optimizer_text += f"lr          :{optimizer.defaults['lr']} \n"
        optimizer_text += f"batchsize   :{args.batchsize} \n"
        optimizer_text += f"weight_decay:{args.weight_decay} \n"
        optimizer_text += f"{args.extra_stuff_you_want_to_add_to_tb}"
        sumwriter.add_text('Parameters', optimizer_text, 0)
    
    # Train loops
    for epoch in range(args.start_epoch, args.num_epochs):
        print(f'Training epoch {epoch}')
        writer = sumwriter if gpu == 0 else None
        train_pcoders(
            pnet, epoch, loss_function, optimizer, writer, train_loader, args)
        if gpu == 0:
            test_pcoders(pnet, epoch, loss_function, optimizer, writer, val_loader, args)
            for pcod_idx in range(NUMBER_OF_PCODERS):
                torch.save({
                    'pcoderweights':getattr(pnet,f"pcoder{pcod_idx+1}").state_dict(),
                    'optimizer'    :optimizer.state_dict(),
                    'epoch'        :epoch,
                    }, f'{args.pth_dir}pnet_pretrained_pc{pcod_idx+1}_{epoch:03d}.pth')


#####################################################
#              BEGIN SCRIPT
#####################################################
if __name__ == '__main__':
    args = Args()
    free_port = get_open_port()
    
    if args.random_seed:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pth_dir, exist_ok=True)
    
    train_subset_indices = np.random.choice(1281167, size=80000, replace=False)
    val_subset_indices = np.random.choice(50000, size=9000, replace=False)
    
    mp_args = {}
    mp_args['train_subset_indices'] = train_subset_indices
    mp_args['val_subset_indices'] = val_subset_indices
    mp_args['free_port'] = free_port
    mp_args['args'] = args
    mp_args['NUMBER_OF_PCODERS'] = NUMBER_OF_PCODERS
    mp.spawn(train_and_eval, nprocs=args.num_gpus, args=(mp_args,))
    
