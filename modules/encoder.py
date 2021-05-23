import torch

from utils.model_utils import build_dense_nn


class Encoder(torch.nn.Module):
    def __init__(self, z_dim, shared_units, separate_units, use_batch_norm=False):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = build_dense_nn(784, shared_units,
                                  use_batch_norm=use_batch_norm,
                                  last_batchnorm=True,
                                  last_activation='relu')

        self.fc21 = build_dense_nn(shared_units[-1], separate_units + [z_dim],
                                  use_batch_norm=use_batch_norm,
                                  last_batchnorm=False,
                                  last_activation=None)
        self.fc22 = build_dense_nn(shared_units[-1], separate_units + [z_dim],
                                  use_batch_norm=use_batch_norm,
                                  last_batchnorm=False,
                                  last_activation=None)

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.fc1(x)
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale