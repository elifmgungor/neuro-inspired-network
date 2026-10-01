'''
Network architectures.
All models share the same CNN; only the hidden-layer neurons differ:
    - baseline:        standard neurons, relu(sum of weighted inputs)
    - linear_branches: inputs are split into dendritic branches, branches are summed linearly (control)
    - dendritic:       same branches, but each branch applies its own ReLU before the soma sums them
    - dale:            standard neurons that are either excitatory or inhibitory (Dale's law)
    - divisive_norm:   standard neurons whose responses are divided by the activity of the whole layer
'''
import math

import torch
from torch import nn

MODEL_NAMES = ['baseline', 'linear_branches', 'dendritic', 'dale', 'divisive_norm']
N_FEATURES = 64 * 8 * 8  # size of the flattened output of the convolution blocks

class DendriticLayer(nn.Module):
    '''
    Layer of neurons with dendritic branches.
    The inputs are shuffled once with a fixed random order and split into n_branches equal groups.
    Every neuron has one branch per group:
        branch_b = f(w_b · x_b + c_b)       f = ReLU if branch_nonlinearity else identity
        output   = sum_b s_b * branch_b + d  (soma)
    Each neuron still receives all inputs, so the only difference between the
    dendritic model and the linear_branches control is the branch nonlinearity f.
    '''
    def __init__(self, n_inputs, n_neurons, n_branches, branch_nonlinearity):
        super().__init__()
        self.n_branches = n_branches
        self.branch_size = n_inputs // n_branches
        self.branch_nonlinearity = branch_nonlinearity

        # Fixed random assignment of inputs to branches (saved with the model, not trained)
        self.register_buffer('input_order', torch.randperm(n_inputs))

        # Same uniform initialization as nn.Linear: +-1/sqrt(number of inputs)
        branch_bound = 1 / math.sqrt(self.branch_size)
        soma_bound = 1 / math.sqrt(n_branches)
        self.branch_weights = nn.Parameter(torch.empty(n_neurons, n_branches, self.branch_size).uniform_(-branch_bound, branch_bound))
        self.branch_bias = nn.Parameter(torch.empty(n_neurons, n_branches).uniform_(-branch_bound, branch_bound))
        self.soma_weights = nn.Parameter(torch.empty(n_neurons, n_branches).uniform_(-soma_bound, soma_bound))
        self.soma_bias = nn.Parameter(torch.empty(n_neurons).uniform_(-soma_bound, soma_bound))

    def forward(self, x):
        # 1. Split the inputs into branches: (batch, n_branches, branch_size)
        x = x[:, self.input_order].reshape(len(x), self.n_branches, self.branch_size)

        # 2. Each neuron's branches: (batch, n_neurons, n_branches)
        branches = torch.einsum('zbk,nbk->znb', x, self.branch_weights) + self.branch_bias
        if self.branch_nonlinearity:
            branches = torch.relu(branches)

        # 3. Soma combines its branches: (batch, n_neurons)
        return (branches * self.soma_weights).sum(dim=2) + self.soma_bias

class DaleLinear(nn.Module):
    '''
    Linear layer whose inputs follow Dale's law: every input neuron is either
    excitatory (all its outgoing weights >= 0) or inhibitory (all <= 0).
    The first excitatory_fraction of the input neurons are excitatory, the rest inhibitory.
    The network learns the size of each weight; the sign is fixed.
    '''
    def __init__(self, n_inputs, n_outputs, excitatory_fraction=0.8):
        super().__init__()
        # Start from the same random weights as nn.Linear, then keep only their size
        linear = nn.Linear(n_inputs, n_outputs)
        self.weight_size = nn.Parameter(linear.weight.detach().abs())
        self.bias = nn.Parameter(linear.bias.detach())

        n_excitatory = round(excitatory_fraction * n_inputs)
        sign = torch.ones(n_inputs)
        sign[n_excitatory:] = -1
        self.register_buffer('sign', sign)

    def forward(self, x):
        weight = self.weight_size.abs() * self.sign  # column j gets the sign of input neuron j
        return x @ weight.T + self.bias

class DivisiveNormalization(nn.Module):
    '''
    Divisive normalization (Carandini & Heeger, 2012):
        r_i = h_i / (sigma + mean_j h_j)
    Each neuron's response is divided by the mean activity of the layer.
    sigma is learned and kept positive.
    '''
    def __init__(self):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def forward(self, h):
        return h / (torch.exp(self.log_sigma) + h.mean(dim=1, keepdim=True))

class CNN(nn.Module):
    '''
    Small CNN: two convolution blocks followed by one hidden layer and the output layer.
    Input: 3 x 32 x 32 CIFAR-10 image. Output: 10 class scores.
    model_name chooses the hidden-layer neurons (see MODEL_NAMES).
    '''
    def __init__(self, model_name, n_hidden=128, n_branches=8):
        super().__init__()
        # The layers are always created in the same order, so with the same seed
        # the convolution layers start from the same weights in every model.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # Hidden layer
        if model_name == 'linear_branches':
            self.fc1 = DendriticLayer(N_FEATURES, n_hidden, n_branches, branch_nonlinearity=False)
        elif model_name == 'dendritic':
            self.fc1 = DendriticLayer(N_FEATURES, n_hidden, n_branches, branch_nonlinearity=True)
        else:
            self.fc1 = nn.Linear(N_FEATURES, n_hidden)

        # Optional normalization of the hidden layer
        if model_name == 'divisive_norm':
            self.normalization = DivisiveNormalization()
        else:
            self.normalization = nn.Identity()

        # Output layer (Dale's law constrains the outgoing weights of the hidden neurons)
        if model_name == 'dale':
            self.fc2 = DaleLinear(n_hidden, 10)
        else:
            self.fc2 = nn.Linear(n_hidden, 10)

        if model_name not in MODEL_NAMES:
            raise ValueError(f'Unknown model: {model_name}')

    def hidden(self, x):
        '''
        Hidden-layer activity for a batch of images: (batch, n_hidden).
        '''
        # 1. Convolution blocks: 32x32 -> 16x16 -> 8x8
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # 2. Hidden layer (the only part that differs between models)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.normalization(x)

    def forward(self, x):
        # 3. Class scores
        return self.fc2(self.hidden(x))
