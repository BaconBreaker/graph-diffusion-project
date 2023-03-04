"""
Functions for loading and preprocessing data

@Author Thomas Chirstensen and Rasmus Pallisgaard
"""
import torch
import torchvision
from torch.utils.data import DataLoader

import os


def get_data(args, subset=None):
    if args.graph_data:
        return get_data_graph(args, subset)
    else:
        return get_data_(args, subset)


def get_data_graph(args, subset=None):
    adjecency_matrix = torch.load(os.path.join(args.dataset_path, "adjecency_tensor_train.pt"))
    node_features = torch.load(os.path.join(args.dataset_path, "node_tensor_train.pt"))
    pdfs = torch.load(os.path.join(args.dataset_path, "pdf_tensor_train.pt"))
    if subset is not None:
        adjecency_matrix = adjecency_matrix[:subset]
        node_features = node_features[:subset]
        pdfs = pdfs[:subset]

    # This is just to make it fit into the image setup.
    adjecency_matrix = adjecency_matrix[:, None, :134, :134]
    node_features = node_features[:, :134, :]
    labels_tmp = torch.ones(adjecency_matrix.shape[0], dtype=torch.long)
    pad_mask = torch.ones(node_features.shape[0:2], dtype=bool)

    dataset = torch.utils.data.TensorDataset(adjecency_matrix, labels_tmp, node_features, pdfs, pad_mask)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataloader


def get_data_(args, subset=None):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(88),
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    if subset is not None:
        dataset = torch.utils.data.Subset(dataset, range(subset))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader
