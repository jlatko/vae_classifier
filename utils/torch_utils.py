import torch

USE_CUDA = torch.cuda.is_available()

def to_gpu(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x