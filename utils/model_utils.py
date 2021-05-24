from torch import nn

def build_dense_nn(input_dim, units, use_batch_norm=False, last_batchnorm=False, last_activation=None):
    layers = []
    next_input = input_dim

    for u in units[:-1]:
        layers.append(nn.Linear(next_input, u))
        next_input = u
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(u))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(next_input, units[-1]))

    if use_batch_norm and last_batchnorm:
        layers.append(nn.BatchNorm1d(units[-1]))

    if last_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif last_activation == "relu":
        layers.append(nn.ReLU())
    elif last_activation == "softplus":
        layers.append(nn.ReLU())
    elif last_activation == "tanh":
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)