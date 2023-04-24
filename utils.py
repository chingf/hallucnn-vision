import numpy as np
import torch

def to_pair(input):
    if isinstance(input, tuple):
        return input[:2]
    else:
        return input, input
    
class PhaseShuffle(object):
    r"""
    Shuffles phase info of image
    """
    def __init__(self):
        pass
        
    def __call__(self, tensor):
        
        y = torch.fft.rfft2(tensor)

        res = np.real(y)
        ims = np.imag(y)
        ims_shape = ims.shape
        shuff_indices = np.arange(ims_shape[-1]*ims_shape[-2])
        np.random.shuffle(shuff_indices)
        for c in [1]:
            c_vals = ims[c]
            c_shape = c_vals.shape
            c_vals = c_vals.flatten()
            c_vals = c_vals[shuff_indices]
            ims[c] = c_vals.reshape(c_shape)
        new_image = res +ims*1j
        return torch.fft.irfft2(new_image) 
    
    def __repr__(self):
        return self.__class__.__name__ 
    
class MagShuffle(object):
    r"""
    Shuffles magnitude info of image
    """
    def __init__(self):
        pass
        
    def __call__(self, tensor):
        
        y = torch.fft.rfft2(tensor)

        res = np.real(y)
        ims = np.imag(y)
        res_shape = res.shape
        shuff_indices = np.arange(res_shape[-1]*res_shape[-2])
        np.random.shuffle(shuff_indices)
        for c in [1]:
            c_vals = res[c]
            c_shape = c_vals.shape
            c_vals = c_vals.flatten()
            c_vals = c_vals[shuff_indices]
            res[c] = c_vals.reshape(c_shape)
        new_image = res +ims*1j
        return torch.fft.irfft2(new_image) 
    
    def __repr__(self):
        return self.__class__.__name__ 

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
