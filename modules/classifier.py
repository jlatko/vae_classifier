import torch

from utils.model_utils import build_dense_nn


class Classifier(torch.nn.Module):
    def __init__(self, units, use_batch_norm=False):
        super().__init__()
        # setup the three linear transformations used
        self.nn = build_dense_nn(784, units + [10],
                                 use_batch_norm=use_batch_norm,
                                 last_batchnorm=False,
                                 last_activation=None)

    def forward(self, x):
        x = x.reshape(-1, 784)
        return self.nn(x) # logits
