#!/usr/bin/env python3
"""
Preprocesses the data into fully connected graphs with node features and edge features

args:
    data_dir: Directory containing the data
    save_dir: Directory to save the processed data

@Author: Thomas Christensen
"""

import argparse
import torch
import h5py
import glob
from tqdm import tqdm
import numpy as np
torch.manual_seed(0)  # Reproducability


def load_structure(path):

    with h5py.File(path, 'r') as file:
        edge_features = torch.tensor(file['edge_attributes'][:], dtype=torch.float32)  # Edge attributes
        edge_indices = torch.tensor(file['edge_indices'][:], dtype=torch.long)  # Edge (sparse) adjecency matrix
        node_features = torch.tensor(file['node_attributes'][:], dtype=torch.float32)  # Node attributes

        r = file['r'][...]  # Real space (x-axis)
        pdf = file['pdf'][...]  # G(r) (y-axis)

    return edge_features, edge_indices, node_features, r, pdf


def pad_nodes(array, max_length):
    """Adds padding over axis 0 to array"""
    padded_array = torch.zeros((max_length, *array.shape[1:]))
    padded_array[:array.shape[0]] = array
    return padded_array


def make_adjecency_matrix(edge_indices, edge_features, max_length):
    """Creates a dense adjecency matrix from a sparse adjecency matrix"""
    adjecency_matrix = torch.zeros((max_length, max_length))
    adjecency_matrix[edge_indices[0, :], edge_indices[1, :]] = edge_features
    return adjecency_matrix


def train_test_split(node_tensor, adjecency_tensor, pdf_tensor, num_particles, test_size=0.1, val_size=0.1):
    """Splits the data into training, validation and test sets"""
    # Shuffle the data
    perm = torch.randperm(node_tensor.shape[0])
    node_tensor = node_tensor[perm]
    adjecency_tensor = adjecency_tensor[perm]
    pdf_tensor = pdf_tensor[perm]
    num_particles = num_particles[perm]

    # Split the data
    test_size = int(node_tensor.shape[0] * test_size)
    val_size = int(node_tensor.shape[0] * val_size)
    train_size = node_tensor.shape[0] - test_size - val_size

    train_node_tensor = node_tensor[:train_size]
    train_adjecency_tensor = adjecency_tensor[:train_size]
    train_pdf_tensor = pdf_tensor[:train_size]
    train_num_particles = num_particles[:train_size]

    val_node_tensor = node_tensor[train_size:train_size+val_size]
    val_adjecency_tensor = adjecency_tensor[train_size:train_size+val_size]
    val_pdf_tensor = pdf_tensor[train_size:train_size+val_size]
    val_num_particles = num_particles[train_size:train_size+val_size]

    test_node_tensor = node_tensor[train_size+val_size:]
    test_adjecency_tensor = adjecency_tensor[train_size+val_size:]
    test_pdf_tensor = pdf_tensor[train_size+val_size:]
    test_num_particles = num_particles[train_size+val_size:]

    return train_node_tensor, train_adjecency_tensor, train_pdf_tensor, train_num_particles, \
        val_node_tensor, val_adjecency_tensor, val_pdf_tensor, val_num_particles, \
        test_node_tensor, test_adjecency_tensor, test_pdf_tensor, test_num_particles


def save_data(save_dir, node_tensor, adjecency_tensor, pdf_tensor, num_particles, r, split):
    """Saves the data to a .pt file"""
    torch.save(node_tensor, f'{save_dir}/node_tensor_{split}.pt')
    torch.save(num_particles, f'{save_dir}/num_particles_{split}.pt')
    torch.save(adjecency_tensor, f'{save_dir}/adjecency_tensor_{split}.pt')
    torch.save(r, f'{save_dir}/r_tensor.pt')  # r is the same for all graphs
    torch.save(pdf_tensor, f'{save_dir}/pdf_tensor_{split}.pt')


def main(args):
    file_names = glob.glob(f'{args.data_dir}/*.h5')

    # Max number of nodes = 511
    # Number of node attributes = 7
    node_tensor = []
    adjecency_tensor = []
    pdf_tensor = []
    num_particles = []

    for p in tqdm(file_names):
        edge_features, edge_indices, node_features, r, pdf = load_structure(p)

        if node_features.shape[0] != 135:  # Only take size 135 graphs
            continue

        node_tensor.append(node_features)
        adjecency_tensor.append(make_adjecency_matrix(edge_indices, edge_features, 135))
        pdf_tensor.append(pdf)
        num_particles.append(int(node_features.shape[0]))

    node_tensor = torch.from_numpy(np.stack(node_tensor))
    adjecency_tensor = torch.from_numpy(np.stack(adjecency_tensor))
    pdf_tensor = torch.from_numpy(np.stack(pdf_tensor))
    num_particles = torch.tensor(num_particles)

    print("sucessfully loaded {} graphs".format(node_tensor.shape[0]))

    # Split the data into training, validation and test sets
    train_node_tensor, train_adjecency_tensor, train_pdf_tensor, train_num_particles, \
        val_node_tensor, val_adjecency_tensor, val_pdf_tensor, val_num_particles, \
        test_node_tensor, test_adjecency_tensor, test_pdf_tensor, test_num_particles = train_test_split(
            node_tensor, adjecency_tensor, pdf_tensor, num_particles)

    # Save the data
    save_data(f'{args.save_dir}/',
              train_node_tensor, train_adjecency_tensor, train_pdf_tensor, train_num_particles, r, 'train')
    save_data(f'{args.save_dir}/',
              val_node_tensor, val_adjecency_tensor, val_pdf_tensor, val_num_particles, r, 'val')
    save_data(f'{args.save_dir}/',
              test_node_tensor, test_adjecency_tensor, test_pdf_tensor, test_num_particles, r, 'test')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='graphs_h5/')
    parser.add_argument('--save_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
