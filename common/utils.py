import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from torch.optim.lr_scheduler import *
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional as torchvision_F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict


def load_CIFAR10_dataset(batch_size: int = 128, calibration_batch_size: int = 1024,
                         data_path: str = './data'):
    """
    download and loading the data loaders
    Args:
        batch_size: batch size for train and test loader
        calibration_batch_size: size of the calibration batch
        data_path: directory to save data

    Returns:
        train_loader, test_loader, calibration_loader
    """
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    num_workers = os.cpu_count()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_data = dset.CIFAR10(data_path,
                              train=True,
                              transform=train_transform,
                              download=True)
    test_data = dset.CIFAR10(data_path,
                             train=False,
                             transform=test_transform,
                             download=True)
    calibration_data = dset.CIFAR10(data_path,
                                    train=True,
                                    transform=test_transform,
                                    download=False)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    train_idx, calibration_idx = indices[calibration_batch_size:], indices[:calibration_batch_size]
    train_sampler = SubsetRandomSampler(train_idx)
    calibration_sampler = SubsetRandomSampler(calibration_idx)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    calibration_loader = DataLoader(
        calibration_data,
        batch_size=calibration_batch_size,
        sampler=calibration_sampler,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, test_loader, calibration_loader


def show_samples(test_data):
    """
    plot 4 samples of each classes in CIFAR10
    Args:
        test_data:

    Returns:

    """
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    samples = [[] for _ in range(10)]
    for image, label in test_data:
        if len(samples[label]) < 4:
            samples[label].append(image)

    fig, axes = plt.subplots(4, 10, squeeze=False, figsize=(10 * 3, 4 * 3))
    for i in range(10):
        for j in range(4):
            img = samples[i][j].detach()
            for c in range(img.shape[0]):
                img[c] = img[c] * std[c] + mean[c]
            img = torchvision_F.to_pil_image(img)

            axes[j, i].imshow(np.asarray(img))
            axes[j, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axes[j, i].set_title(test_data.classes[i])


def train(
        model: nn.Module,
        dataflow: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        device: torch.device = torch.device("cuda")
) -> None:
    model.train()

    for inputs, targets in dataflow:
        # Move the data from CPU to GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Reset the gradients (from the last iteration)
        optimizer.zero_grad()

        # Forward inference
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward propagation
        loss.backward()

        # Update optimizer and LR scheduler
        optimizer.step()
        if scheduler is not None:
            scheduler.step()


@torch.inference_mode()
def evaluate(
        model: nn.Module,
        dataflow: DataLoader,
        device: torch.device = torch.device("cuda")
) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in dataflow:
        # Move the data from CPU to GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()


def fit(model: nn.Module, num_epochs: int, train_loader: DataLoader, test_loader: DataLoader,
        criterion: nn.Module, optimizer: Optimizer, scheduler: LambdaLR, device: torch.device = torch.device("cuda")):
    test_accuracy = []
    train_accuracy = []
    for epoch_num in tqdm(range(1, num_epochs + 1), desc="fit", leave=False):
        train(model, train_loader, criterion, optimizer, scheduler, device)
        metric = evaluate(model, train_loader, device)
        train_accuracy.append(metric)
        metric = evaluate(model, test_loader, device)
        test_accuracy.append(metric)
        print(f"epoch {epoch_num}: train_accuracy={train_accuracy[-1]}, test_accuracy={test_accuracy[-1]}")

    return train_accuracy, test_accuracy


from functools import reduce
def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)

def model_size(model):
    param_size = 0
    for name, p in model.named_parameters():
        if 'conv' in name or 'fc' in name:
            if name[-6:] == 'weight':
                m = get_module_by_name(model, name[:-7])
                if m.method == 'normal':
                    element_size = 4
                else:
                    element_size = m.weight_N_bits / 8
                param_size_ = p.nelement() * element_size
            else:
                param_size_ = p.nelement() * p.element_size()
        else:
            param_size_ = p.nelement() * p.element_size()
        param_size += param_size_

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return('model size: {:.3f}MB'.format(size_all_mb))
