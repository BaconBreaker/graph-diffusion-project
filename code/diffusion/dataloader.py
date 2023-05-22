"""
Script conatining dataloader object for training and testing as well as the
function to get the dataloader given args

@Author Thomas Christensen and Rasmus Pallisgaard
"""
import glob
from os import path

import pytorch_lightning as pl
import torch
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils.data import load_structure, pad_array  # make_adjecency_matrix
from utils.graph_transformer_utils import MoleculeDatasetInfo

# Set path to single molecule for debugging
single_sample_name = "graph_AntiFlourite_Ra2O_r5.h5"


class MoleculeDataset(Dataset):
    def __init__(self, sample_list, pad_length, transform=None, cubic=False):
        """
        Dataset class for molecules
        args:
            sample_list: list of paths to samples
            transform: transform to apply to samples
        """
        self.sample_list = sample_list
        self.pad_length = pad_length
        self.transform = transform
        self.cubic = cubic

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
                pdf: (3000) tensor containing the pair distribution function values
                pad_mask: (pad_length) tensor containing a mask for the padding
        """
        sample_path = self.sample_list[idx]
        if not self.cubic:
            edge_features, edge_indices, node_features, r, pdf = load_structure(sample_path)
            n_atoms = node_features.shape[0]
            node_features = pad_array(node_features, self.pad_length, 0)

            xyz = node_features[:, 4:7]
        else:
            n_atoms = 8
            xyz = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0],
                                [0, 1, 1], [1, 0, 0], [1, 0, 1],
                                [1, 1, 0], [1, 1, 1]], dtype=torch.float32)
            node_features = torch.zeros((8, 6))
            r = torch.zeros(3000)
            pdf = torch.zeros(3000)

        pad_mask = torch.zeros(self.pad_length, dtype=torch.bool)
        pad_mask[n_atoms:] = True

        adj_matrix = torch.zeros((self.pad_length, self.pad_length), dtype=torch.float32)
        tri_cor = torch.triu_indices(node_features.shape[0], node_features.shape[0], offset=1)
        distances = torch.tensor(pdist(xyz, metric='euclidean'), dtype=torch.float32)

        # Set the entries of the adjecency matrix to the entries of the distance matrix
        adj_matrix[tri_cor[0], tri_cor[1]] = distances
        adj_matrix[tri_cor[1], tri_cor[0]] = distances

        sample_dict = {
            'adj_matrix': adj_matrix,
            'node_features': node_features,
            'r': r,
            'pdf': pdf,
            'pad_mask': pad_mask}

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)
        return sample_dict


class MoleculeDataModule(pl.LightningDataModule):
    def __init__(self, args, transform):
        """
        Pytorch Lightning datamodule for loading data for training and testing
        args:
            args: argparse object
        """
        super().__init__()
        self.model_type = args.model
        self.dataset_path = args.dataset_path
        self.val_split = args.val_split
        self.num_train_samples = args.num_train_samples
        self.num_val_samples = args.num_val_samples
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.tensors_to_diffuse = args.tensors_to_diffuse
        self.single_sample = args.single_sample
        self.cubic = args.cubic

        self.pad_length = args.pad_length
        self.transform = transform

        self.molecule_info = MoleculeDatasetInfo(n_nodes=args.pad_length,
                                                 n_node_types=2,
                                                 n_edge_types=30)

    def pre_setup(self):
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

        # If overfitting on single sample, set sample paths to single sample
        if self.single_sample:
            self.train_sample_paths = [path.join(self.dataset_path, single_sample_name)] * 128 * self.batch_size
            self.val_sample_paths = [path.join(self.dataset_path, single_sample_name)] * self.batch_size

    def setup(self, stage=None):
        self.pre_setup()

        if stage == "fit" or stage is None:
            self.train_dataset = MoleculeDataset(self.train_sample_paths,
                                                 self.pad_length,
                                                 self.transform,
                                                 cubic=self.cubic)
            self.val_dataset = MoleculeDataset(self.val_sample_paths,
                                               self.pad_length,
                                               self.transform,
                                               cubic=self.cubic)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, dict):
            for n in self.tensors_to_diffuse:
                x = batch[n]
                x = x.to(device)
                batch[n] = x
            if self.model_type == "self_attention":
                batch["pad_mask_sequence"] = batch["pad_mask_sequence"].to(device)
            if self.model_type == "equivariant":
                batch["pdf"] = batch["pdf"].to(device)
                batch["pad_mask"] = batch["pad_mask"].to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)

        return batch
