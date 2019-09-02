import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from config.defaults import cfg

Image.LOAD_TRUNCATED_IMAGES = True


# Transformers
transformer= transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])



def fetch_dataloaders(data_dir):

    # Download and split datasets
    dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transformer)

    trainset, validset = random_split(dataset, (int(len(dataset)*0.7), int(len(dataset)*0.3)))
    testset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transformer)

    # Load dataloaders
    dataloaders = {}
    dataloaders['train'] = DataLoader(trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
    dataloaders['valid'] = DataLoader(validset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
    dataloaders['test']  = DataLoader(testset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)

    return dataloaders


