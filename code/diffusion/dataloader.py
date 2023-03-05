"""
Script conatining dataloader object for training and testing as well as the
function to get the dataloader given args

@Author Thomas Christensen and Rasmus Pallisgaard
"""
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, Dataset

import glob

from utils.data import load_structure, pad_array, make_adjecency_matrix


class MoleculeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Dataset class for molecules
        args:
            sample_list: list of paths to samples
            transform: transform to apply to samples
        """
        self.sample_list = sample_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        """
        Loads sample from disk, constructs tensors and returns them
        """
        sample_path = self.sample_list[idx]
        edge_features, edge_indices, node_features, r, pdf = load_structure(sample_path)
        adj_matrix = make_adjecency_matrix(edge_indices, edge_features, self.pad_length)
        node_features = pad_array(node_features, self.pad_length)

        return adj_matrix, node_features, r, pdf


class MoleculeDataModule(pl.LightningDataModule):
    def __init__(self, args):
        """
        Pytorch Lightning datamodule for loading data for training and testing
        args:
            args: argparse object
        """
        super().__init__()
        self.dataset_path = args.dataset_path
        self.val_split = args.val_split
        self.num_train_samples = args.num_train_samples
        self.num_val_samples = args.num_val_samples
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.pad_length = args.pad_length
        self.transform = None  # TODO: Add transforms

    def prepare_data(self):
        """
        Prepare data for training and testing, only called once on 1 process
        """
        # Collect sample paths and split them into train and val
        all_sample_paths = glob.glob(self.dataset_path + "/*.g5")
        train_sample_paths, val_sample_paths = random_split(
            all_sample_paths, [1 - self.val_split, self.val_split], generator=torch.Generator().manual_seed(42))

        # Set number of samples to use
        num_train = min(self.num_train_samples, len(train_sample_paths))
        num_val = min(self.num_val_samples, len(val_sample_paths))

        # Set sample paths to use
        self.train_sample_paths = train_sample_paths[:num_train]
        self.val_sample_paths = val_sample_paths[:num_val]

        # Save tensors to disk
        self.prepare_tensors(self.train_sample_paths, self.dataset_path + "/train")
        self.prepare_tensors(self.val_sample_paths, self.dataset_path + "/val")

    def prepare_tensors(self, sample_list, save_path):
        """
        Convert data from graph representation to tensor representation
        args:
            sample_list: list of paths to samples
            save_path: path to save tensors to
        """
        adj_matrix_list = []
        node_features_list = []
        r_list = []
        pdf_list = []
        for sample_path in sample_list:
            edge_features, edge_indices, node_features, r, pdf = load_structure(sample_path)
            adj_matrix = make_adjecency_matrix(edge_indices, edge_features, self.pad_length)
            node_features = pad_array(node_features, self.pad_length)

            adj_matrix_list.append(adj_matrix)
            node_features_list.append(node_features)
            r_list.append(r)
            pdf_list.append(pdf)

        adj_matrix_tensor = torch.tensor(adj_matrix_list)
        node_features_tensor = torch.tensor(node_features_list)
        r_tensor = torch.tensor(r_list)
        pdf_tensor = torch.tensor(pdf_list)

        torch.save(adj_matrix_tensor, save_path + "/adj_matrix.pt")
        torch.save(node_features_tensor, save_path + "/node_features.pt")
        torch.save(r_tensor, save_path + "/r.pt")
        torch.save(pdf_tensor, save_path + "/pdf.pt")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MoleculeDataset(self.train_sample_paths, self.pad_length, self.transform)
            self.val_dataset = MoleculeDataset(self.val_sample_paths, self.pad_length, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size,
                          shuffle=False, num_workers=self.num_workers)
