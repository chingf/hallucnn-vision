#!/usr/bin/env python
# coding: utf-8

# In[27]:


import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from models.networks_2022 import BranchedNetwork
from models.pbranchednetwork_all import PBranchedNetwork_AllSeparateHP


# In[28]:


engram_dir = '/mnt/smb/locker/abbott-locker/hcnn/'


# # 1. Random network with default initialization
# Kaiming uniform is the PyTorch default

# In[3]:


random_name = 'untrained_random_kaiming'
random_chckpt_dir = f'{engram_dir}1_checkpoints/{random_name}/'
os.makedirs(random_chckpt_dir, exist_ok=True)


# In[29]:


net = BranchedNetwork()
net.load_state_dict(torch.load(f'{engram_dir}networks_2022_weights.pt'))


# In[5]:


for i in range(7):
    pnet = PBranchedNetwork_AllSeparateHP(
        net, build_graph=True, random_init=False
        )
    save_path = f'{random_chckpt_dir}'
    save_path += f'{random_name}-{i}-regular.pth'
    torch.save(pnet.state_dict(), save_path)


# # 2. Random network with distribution-match

# ### Set up destination parameters

# In[3]:


net_name = 'untrained_random_matched'
dest_chckpt_dir = f'{engram_dir}1_checkpoints/{net_name}/'
os.makedirs(dest_chckpt_dir, exist_ok=True)


# ### Get source weights

# In[4]:


source_net_name = 'pnet'
source_net_chckpt = 1960
source_chckpt_dir = f'{engram_dir}1_checkpoints/{source_net_name}/'
source_state_dict_path = f'{source_chckpt_dir}{source_net_name}-{source_net_chckpt}-regular.pth'


# In[5]:


source_state_dict = torch.load(
    source_state_dict_path, map_location=torch.device('cpu'))


# In[15]:


bias_distrib = np.concatenate((
    source_state_dict['pcoder1.pmodule.1.bias'].numpy(),
    source_state_dict['pcoder2.pmodule.1.bias'].numpy(),
    source_state_dict['pcoder3.pmodule.1.bias'].numpy(),
    source_state_dict['pcoder4.pmodule.0.bias'].numpy(),
    source_state_dict['pcoder5.pmodule.0.bias'].numpy()
    ))


# In[26]:


for i in range(7):
    new_state_dict = torch.load(
        source_state_dict_path, map_location=torch.device('cpu'))
    for bias_name in [
        'pcoder1.pmodule.1.bias', 'pcoder2.pmodule.1.bias',
        'pcoder3.pmodule.1.bias', 'pcoder4.pmodule.0.bias',
        'pcoder5.pmodule.0.bias']:
        new_state_dict[bias_name] = torch.tensor(np.random.choice(
            bias_distrib, size=source_state_dict[bias_name].shape
            ))

    for weight_name in [
        'pcoder1.pmodule.1.weight', 'pcoder2.pmodule.1.weight',
        'pcoder3.pmodule.1.weight', 'pcoder4.pmodule.0.weight',
        'pcoder5.pmodule.0.weight'
        ]:
        source_distrib = source_state_dict[weight_name].flatten().numpy()
        new_state_dict[weight_name] = torch.tensor(np.random.choice(
            source_distrib, size=source_state_dict[weight_name].shape
            ))
    save_path = f'{dest_chckpt_dir}'
    save_path += f'{net_name}-{i}-regular.pth'
    torch.save(new_state_dict, save_path)


# In[ ]:




