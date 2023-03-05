"""
Functions for loading and preprocessing data

@Author Thomas Chirstensen and Rasmus Pallisgaard
"""
import torch
import torchvision
from torch.utils.data import DataLoader

import os
import h5py


def get_data(args, subset=None):
    if args.graph_data:
        return get_data_graph(args, subset)
    else:
        return get_data_(args, subset)


def get_data_graph(args, subset=None):
    """
    OBSOLETE
    """
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
    """
    OBSOLETE
    """
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


def load_structure(path):
    """
    Loads a structure from a .h5 file as a tuple of tensors.
    """
    with h5py.File(path, 'r') as file:
        edge_features = torch.tensor(file['edge_attributes'][:], dtype=torch.float32)  # Edge attributes
        edge_indices = torch.tensor(file['edge_indices'][:], dtype=torch.long)  # Edge (sparse) adjecency matrix
        node_features = torch.tensor(file['node_attributes'][:], dtype=torch.float32)  # Node attributes

        r = file['r'][...]  # Real space (x-axis)
        pdf = file['pdf'][...]  # G(r) (y-axis)

    return edge_features, edge_indices, node_features, r, pdf


def pad_array(array, max_length):
    """
    Adds padding over axis 0 to array, all padded values are -1
    args:
        array: array to pad
        max_length: length to pad to
    returns:
        padded_array: padded array
    """
    padded_array = torch.zeros((max_length, *array.shape[1:])) - 1
    padded_array[:array.shape[0]] = array
    return padded_array


def make_adjecency_matrix(edge_indices, edge_features, max_length):
    """
    Creates a dense adjecency matrix from a sparse adjecency matrix
    args:
        edge_indices: sparse adjecency matrix of longs
        edge_features: edge attributes of floats
        max_length: length of adjecency matrix
    returns:
        adjecency_matrix: dense adjecency matrix of floats
    """
    adjecency_matrix = torch.zeros((max_length, max_length))
    adjecency_matrix[edge_indices[0, :], edge_indices[1, :]] = edge_features
    return adjecency_matrix
