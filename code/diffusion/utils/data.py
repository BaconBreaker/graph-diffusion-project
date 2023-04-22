"""
Functions for loading and preprocessing data

@Author Thomas Chirstensen and Rasmus Pallisgaard
"""
import torch
import h5py


def load_structure(path):
    """
    Loads a structure from a .h5 file as a tuple of tensors.
    """
    with h5py.File(path, 'r') as file:
        edge_features = torch.tensor(file['edge_attributes'][:], dtype=torch.float32)  # Edge attributes
        edge_indices = torch.tensor(file['edge_indices'][:], dtype=torch.long)  # Edge (sparse) adjecency matrix
        node_features = torch.tensor(file['node_attributes'][:], dtype=torch.float32)  # Node attributes

        r = torch.tensor(file['r'][...], dtype=torch.float32)
        pdf = torch.tensor(file['pdf'][...], dtype=torch.float32)

    return edge_features, edge_indices, node_features, r, pdf


def pad_array(array, max_length, pad_value=0):
    """
    Adds padding over axis 0 to array
    args:
        array: array to pad
        max_length: length to pad to
        pad_value: value to pad with
    returns:
        padded_array: padded array
    """
    padded_array = torch.zeros((max_length, *array.shape[1:])) + pad_value
    print(f"padded_array shape: {padded_array.shape}")
    print(f"padded_array reformatted shape: {padded_array[:array.shape[0]].shape}")
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
    # Once again for symmetry
    adjecency_matrix[edge_indices[1, :], edge_indices[0, :]] = edge_features
    return adjecency_matrix
