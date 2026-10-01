'''
CIFAR-10 data loading.
The official training set is split into train and validation sets.
The official test set is only used for the final evaluation.
'''
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Channel means and standard deviations of the CIFAR-10 training set
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def load_cifar10(data_directory, train_percent=100, batch_size=128, n_validation=5000, random_state=42):
    '''
    Download CIFAR-10 (only the first time) and return train, validation and test loaders.
    train_percent keeps only part of the training set (to test learning with fewer labels).
    The same random_state always gives the same split and the same training subset.
    '''
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    full_train_set = datasets.CIFAR10(data_directory, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(data_directory, train=False, download=True, transform=transform)

    # 1. Split the official training set into train and validation
    n_train = len(full_train_set) - n_validation
    generator = torch.Generator().manual_seed(random_state)
    train_set, validation_set = random_split(full_train_set, [n_train, n_validation], generator=generator)

    # 2. Keep train_percent of the training images (validation and test sets stay the same)
    n_keep = n_train * train_percent // 100
    keep = torch.randperm(n_train, generator=generator)[:n_keep]
    train_set = Subset(train_set, keep.tolist())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, validation_loader, test_loader
