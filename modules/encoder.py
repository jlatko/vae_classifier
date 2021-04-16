import torch

class Encoder(torch.nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = torch.nn.Linear(784, hidden_dim)
        self.fc21 = torch.nn.Linear(hidden_dim, z_dim)
        self.fc22 = torch.nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale