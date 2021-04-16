import torch


class Classifier(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = torch.nn.Linear(784, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 10)
        # setup the non-linearities
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        x = x.reshape(-1, 784)
        hidden = self.softplus(self.fc1(x))
        return self.fc2(hidden)