'''
Network architectures.

CNN models use the same convolutional architecture and initialization, trained independently; only the hidden-layer neurons differ:
    - baseline:               standard neurons, relu(sum of weighted inputs)
    - linear_branches:        inputs are split into dendritic branches, branches are summed linearly (control)
    - dendritic:              same branches, but each branch applies its own ReLU before the soma sums them
    - linear_branches_sparse: like linear_branches, but each branch samples only 16 random inputs
    - dendritic_sparse:       like dendritic, but each branch samples only 16 random inputs
    - dale:                   standard neurons that are either excitatory or inhibitory (Dale's law)
    - divisive_norm:          standard neurons whose responses are divided by the activity of the whole layer

MLP models work directly on the pixels and follow Chavlis & Poirazi (2025):
    - mlp_vanilla:                fully connected, 3072 -> 4096 -> 256 -> 10 (their vANN)
    - mlp_dendritic_sparse:       256 somata x 16 dendrites, each dendrite samples 16 random pixels (their dANN-R)
    - mlp_linear_branches_sparse: same, but without the dendritic nonlinearity (control not in the paper)
'''
import math

import torch
from torch import nn

CNN_MODELS = ['baseline', 'linear_branches', 'dendritic', 'linear_branches_sparse', 'dendritic_sparse',
              'dale', 'divisive_norm']
MLP_MODELS = ['mlp_vanilla', 'mlp_dendritic_sparse', 'mlp_linear_branches_sparse']
MODEL_NAMES = CNN_MODELS + MLP_MODELS

N_FEATURES = 64 * 8 * 8  # size of the flattened output of the convolution blocks
N_PIXELS = 3 * 32 * 32   # size of a flattened CIFAR-10 image

class DendriticLayer(nn.Module):
    '''
    Layer of neurons with dendritic branches. For every neuron:
        branch_b = f(w_b · x_b + c_b)       f = branch_activation (ReLU, LeakyReLU or identity)
        output   = sum_b s_b * branch_b + d  (soma)
    Which inputs x_b a branch receives:
        - inputs_per_branch = None: the inputs are shuffled once and split into n_branches equal groups,
          so every neuron still receives all inputs (the same split for every neuron).
        - inputs_per_branch = k: every branch of every neuron samples its own k random inputs
          (restricted sampling, as in Chavlis & Poirazi, 2025).
    '''
    def __init__(self, n_inputs, n_neurons, n_branches, branch_activation, inputs_per_branch=None):
        super().__init__()
        self.n_branches = n_branches
        self.inputs_per_branch = inputs_per_branch
        self.branch_activation = branch_activation

        # Fixed random assignment of inputs to branches (saved with the model, not trained)
        if inputs_per_branch is None:
            branch_size = n_inputs // n_branches
            self.register_buffer('input_order', torch.randperm(n_inputs))
        else:
            branch_size = inputs_per_branch
            random_order = torch.rand(n_neurons, n_branches, n_inputs).argsort(dim=2)
            self.register_buffer('input_index', random_order[:, :, :inputs_per_branch])
        self.branch_size = branch_size

        # Same uniform initialization as nn.Linear: +-1/sqrt(number of inputs)
        branch_bound = 1 / math.sqrt(branch_size)
        soma_bound = 1 / math.sqrt(n_branches)
        self.branch_weights = nn.Parameter(torch.empty(n_neurons, n_branches, branch_size).uniform_(-branch_bound, branch_bound))
        self.branch_bias = nn.Parameter(torch.empty(n_neurons, n_branches).uniform_(-branch_bound, branch_bound))
        self.soma_weights = nn.Parameter(torch.empty(n_neurons, n_branches).uniform_(-soma_bound, soma_bound))
        self.soma_bias = nn.Parameter(torch.empty(n_neurons).uniform_(-soma_bound, soma_bound))

    def forward(self, x):
        # 1. Each neuron's branches: (batch, n_neurons, n_branches)
        if self.inputs_per_branch is None:
            x = x[:, self.input_order].reshape(len(x), self.n_branches, self.branch_size)
            branches = torch.einsum('zbk,nbk->znb', x, self.branch_weights) + self.branch_bias
        else:
            x = x[:, self.input_index]  # (batch, n_neurons, n_branches, inputs_per_branch)
            branches = (x * self.branch_weights).sum(dim=3) + self.branch_bias

        # 2. Branch nonlinearity (identity for the linear controls)
        branches = self.branch_activation(branches)

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
    model_name chooses the hidden-layer neurons (see CNN_MODELS).
    '''
    def __init__(self, model_name, n_hidden=128, n_branches=8, inputs_per_branch=16):
        super().__init__()
        if model_name not in CNN_MODELS:
            raise ValueError(f'Unknown CNN model: {model_name}')

        # The layers are always created in the same order, so with the same seed
        # the convolution layers start from the same weights in every model.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # Hidden layer
        if 'branches' in model_name or 'dendritic' in model_name:
            branch_activation = nn.ReLU() if model_name.startswith('dendritic') else nn.Identity()
            sampling = inputs_per_branch if model_name.endswith('_sparse') else None
            self.fc1 = DendriticLayer(N_FEATURES, n_hidden, n_branches, branch_activation, sampling)
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

class MLP(nn.Module):
    '''
    Two-hidden-layer network on raw pixels, following Chavlis & Poirazi (2025).
    Every model has the same number of nodes: 4096 dendrites (or first-layer units) and 256 somata.
    All nodes use LeakyReLU(0.1), as in the paper, except the branches of the linear control.
    '''
    def __init__(self, model_name, n_somata=256, n_dendrites=16, inputs_per_dendrite=16):
        super().__init__()
        if model_name not in MLP_MODELS:
            raise ValueError(f'Unknown MLP model: {model_name}')
        self.activation = nn.LeakyReLU(0.1)

        if model_name == 'mlp_vanilla':
            n_first_layer = n_somata * n_dendrites
            self.hidden_layers = nn.Sequential(nn.Linear(N_PIXELS, n_first_layer), self.activation,
                                               nn.Linear(n_first_layer, n_somata))
        else:
            branch_activation = nn.Identity() if model_name == 'mlp_linear_branches_sparse' else self.activation
            self.hidden_layers = DendriticLayer(N_PIXELS, n_somata, n_dendrites, branch_activation, inputs_per_dendrite)

        self.output = nn.Linear(n_somata, 10)

    def hidden(self, x):
        '''
        Somatic activity for a batch of images: (batch, n_somata).
        '''
        return self.activation(self.hidden_layers(x.flatten(1)))

    def forward(self, x):
        return self.output(self.hidden(x))

def build_model(model_name):
    '''
    Create any model in MODEL_NAMES.
    '''
    if model_name in MLP_MODELS:
        return MLP(model_name)
    return CNN(model_name)
