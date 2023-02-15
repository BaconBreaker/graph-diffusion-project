#!/usr/bin/python
"""
Dataloaders for loading graphs and features into pytorch-compatible format

@Author: Thomas Christensen
"""
import torch
import torch_geometric as tg
from constants import atom_name_dict
import glob
import h5py

def load_structure(path):

    with h5py.File(path, 'r') as file:
        edge_features = torch.tensor(file['edge_attributes'][:], dtype=torch.float32) # Edge attributes
        edge_indices = torch.tensor(file['edge_indices'][:], dtype=torch.long) # Edge (sparse) adjecency matrix
        node_features = torch.tensor(file['node_attributes'][:], dtype=torch.float32) # Node attributes

        pdf = file['pdf'][...]

        graph = tg.data.Data(x = node_features, y = pdf, edge_attr = edge_features, edge_index = edge_indices)
    return graph

def check_metal(data, metals):
    """
    Checks if the graph contains a metal atom of the type specified by the user
    """
    #Note that we onlt check the first atom in the graph because they are ordered with oxygen last
    return atom_name_dict[data.x[0,0].item()].lower() in metals

def SimpleGraphDataLoader(data_dir, batch_size=1, shuffle=True, metals=None):
    """
    Simple dataloader returning torch_geometric.data.Data objects
    """
    metals = [metal.lower() for metal in metals] if metals is not None else []
    file_paths = glob.glob(f'{data_dir}/*.h5')
    data_list = [load_structure(path) for path in file_paths]
    data_list_filtered = [data for data in data_list if check_metal(data, metals)]
    return tg.loader.DataLoader(data_list_filtered, batch_size=batch_size, shuffle=shuffle)


class GraphDataLoader(tg.data.InMemoryDataset):
    def __init__(self, root, data_dir, transform=None, pre_transform=None, pre_filter=None, metals=None):
        """
        Dataloader returning torch_geometric.data.Data objects
        TODO: finish this

        Args:
            data_path (str): _description_
            transform (funciton, optional): Transofrmation function. Defaults to None.
            pre_transform (function, optional): pre transformation funciton. Defaults to None.
            pre_filter (function, optional): Filter. Defaults to None.
        """
        super(GraphDataLoader, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_dir = data_dir
        self.metals = metals
    
    @property
    def raw_file_names(self):
        """
        Built-in property for acessing raw file names
        """
        return glob.glob(f'{self.data_dir}/*.h5')
    
    @property
    def processed_file_names(self):
        """
        Built-in property for acessing processed file names
        """
        ['data.pt']
    
    def download(self):
        """
        Built-in function for downloading data.
        In our case data is assumed to be on disk, so this is empty
        """
        pass

    def process(self):
        """
        Built-in function for pre-processing data.
        """
        data_list = [load_structure(path) for path in self.raw_paths]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])