"""
Script conatining dataloader object for training and testing as well as the
function to get the dataloader given args

@Author Thomas Christensen and Rasmus Pallisgaard
"""
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

import glob

from utils.data import load_structure, pad_array, make_adjecency_matrix


class MoleculeDataset(Dataset):
    def __init__(self, sample_list, pad_length, transform=None):
        """
        Dataset class for molecules
        args:
            sample_list: list of paths to samples
            transform: transform to apply to samples
        """
        self.sample_list = sample_list
        self.pad_length = pad_length
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        """
        Loads sample from disk, constructs tensors and returns them
        args:
            idx: index of sample to load
        returns:
            sample_dict: dictionary containing tensors:
                adj_matrix: (pad_length, pad_length) tensor containing the adjecency matrix
                node_features: (pad_length, 7) tensor containing the node features
                r: (3000) tensor containing the real space values
                pf: (3000) tensor containing the pair distribution function values
                pad_mask: (pad_length, 1) tensor containing a mask for the padding
        """
        sample_path = self.sample_list[idx]
        edge_features, edge_indices, node_features, r, pdf = load_structure(sample_path)

        pad_mask = torch.ones((node_features.shape[0], 1))
        pad_mask[node_features.shape[0]:] = 0

        node_features = pad_array(node_features, self.pad_length, 0)
        ajd_matrix = make_adjecency_matrix(edge_indices, edge_features, self.pad_length)

        sample_dict = {
            'adj_matrix': ajd_matrix,
            'node_features': node_features,
            'r': r,
            'pdf': pdf,
            'pad_mask': pad_mask}

        if self.transform is not None:
            return self.transform(sample_dict)
        else:
            return sample_dict


class MoleculeDataModule(pl.LightningDataModule):
    def __init__(self, args, transform):
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
        self.transform = transform

    def prepare_data(self):
        """
        Prepare data for training and testing, only called once on 1 process
        """
        # Collect sample paths and split them into train and val
        all_sample_paths = glob.glob(self.dataset_path + "/*.h5")
        if len(all_sample_paths) == 0:
            raise ValueError(f"No samples found in {self.dataset_path}")
        train_sample_paths, val_sample_paths = train_test_split(
            all_sample_paths, test_size=self.val_split, random_state=42)

        # Set number of samples to use
        if self.num_train_samples is None:
            self.num_train_samples = len(train_sample_paths)
        if self.num_val_samples is None:
            self.num_val_samples = len(val_sample_paths)
        num_train = min(self.num_train_samples, len(train_sample_paths))
        num_val = min(self.num_val_samples, len(val_sample_paths))

        # Set sample paths to use
        self.train_sample_paths = train_sample_paths[:num_train]
        self.val_sample_paths = val_sample_paths[:num_val]

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MoleculeDataset(self.train_sample_paths, self.pad_length, self.transform)
            self.val_dataset = MoleculeDataset(self.val_sample_paths, self.pad_length, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
