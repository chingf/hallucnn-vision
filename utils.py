import numpy as np
import torch

def to_pair(input):
    if isinstance(input, tuple):
        return input[:2]
    else:
        return input, input

class AddGaussianNoise(object):
    r"""
    Adds Gaussian noise to the images (after conversion to Tensors).
    """
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + (torch.randn(tensor.size()) * self.std + self.mean)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddSaltPepperNoise(object):
    r"""
    Adds Salt&Pepper noise to the images (after conversion to Tensors).
    """
    def __init__(self, probability=0.1):
        self.salt   = probability/2
        self.pepper = 1 - self.salt

    def __call__(self, tensor):
        saltNpepper = torch.rand(tensor.shape[-2], tensor.shape[-1]).repeat(3,1,1)
        noisy = tensor.clone()
        
        salt_v   = torch.max(tensor)
        pepper_v = torch.min(tensor)
        noisy = torch.where(saltNpepper >= self.salt, noisy, salt_v)
        noisy = torch.where(saltNpepper <= self.pepper, noisy, pepper_v)
        
        return noisy
    
    def __repr__(self):
        return self.__class__.__name__ + '(probability={0:0.3f})'.format(self.p)
