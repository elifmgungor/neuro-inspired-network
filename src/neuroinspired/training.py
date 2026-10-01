'''
Training and evaluation functions shared by all models,
so every model is trained and evaluated in exactly the same way.
'''
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

def train_one_epoch(model, loader, optimizer, device):
    '''
    Train the model for one pass over the training data.
    Returns the average training loss.
    '''
    model.train()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)

def predict(model, loader, device, noise_std=0):
    '''
    Return the true labels and predicted labels for all images in the loader.
    noise_std > 0 adds Gaussian noise to the (normalized) images before prediction.
    '''
    model.eval()
    true_labels, predicted_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images + noise_std * torch.randn_like(images)
            scores = model(images.to(device))
            true_labels.append(labels.numpy())
            predicted_labels.append(scores.argmax(dim=1).cpu().numpy())

    return np.concatenate(true_labels), np.concatenate(predicted_labels)

def evaluate(model, loader, device, noise_std=0):
    '''
    Calculate accuracy, balanced accuracy and the confusion matrix.
    '''
    true_labels, predicted_labels = predict(model, loader, device, noise_std)
    return {
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'balanced_accuracy': balanced_accuracy_score(true_labels, predicted_labels),
        'confusion_matrix': confusion_matrix(true_labels, predicted_labels),
    }

def get_hidden_activations(model, loader, device):
    '''
    Return the hidden-layer activity (images x neurons) and the true labels for all images in the loader.
    '''
    model.eval()
    activations, labels_list = [], []

    with torch.no_grad():
        for images, labels in loader:
            activations.append(model.hidden(images.to(device)).cpu().numpy())
            labels_list.append(labels.numpy())

    return np.concatenate(activations), np.concatenate(labels_list)
