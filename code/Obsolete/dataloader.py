#!/usr/bin/python
"""
Dataloaders for loading graphs and features into pytorch-compatible format

@Author: Thomas Christensen
"""
import torch
import torch_geometric as tg
from code.Obsolete.constants import atom_name_dict
import glob
import h5py
import sys
import os
sys.path.append('./DiGress')
from DiGress.src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos

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

def filter_check_metal(metals):
    """
    Returns a function that checks if the graph contains a metal atom of the type specified by the user
    """
    metals = [metal.lower() for metal in metals] if metals is not None else []
    return lambda data: check_metal(data, metals)

def discreteize_edge_features(data):
    """
    Discreteize edge features as 0 or 1
    """
    edge_attr = data.edge_attr
    edge_attr[edge_attr.nonzero(as_tuple=True)] = 1.0
    data.edge_attr = edge_attr
    return data

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
    def __init__(self, root, data_dir, transform=None, pre_transform=None, pre_filter=None, pre_filter_path=None, metals=None):
        """
        Dataloader that follows the python torch_geometric.data.InMemoryDataset format

        Args:
            data_path (str): _description_
            transform (funciton, optional): Transofrmation function. Defaults to None.
            pre_transform (function, optional): pre transformation funciton. Defaults to None.
            pre_filter (function, optional): Filter. Defaults to None.
            pre_filter_path (function, optional): 
        """
        self.data_dir = data_dir
        super(GraphDataLoader, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.metals = metals
        self.root = root

    @property
    def raw_file_names(self):
        """
        Built-in property for acessing raw file names
        """
        return os.listdir(self.data_dir)
    
    @property
    def processed_file_names(self):
        """
        Built-in property for acessing processed file names
        """
        return ['train.pt', 'val.pt', 'test.pt']
    
    def download(self):
        """
        Built-in function for downloading data.
        In this case we just copy it from the data_dir
        """
        os.system(f'cp -r {self.data_dir}/* {self.raw_dir}/')

    def process(self):
        """
        Built-in function for pre-processing data.
        """
        all_paths = self.raw_paths
        
        # filter data based on paths
        
        # Divide data into train, val and test
        
        data_list = [load_structure(path) for path in all_paths]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]        

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class customDataModule(MolecularDataModule):
    pass

class customInfo(AbstractDatasetInfos):
    pass

if __name__ == '__main__':
    # Test if it works
    root = '/home/thomas/tmp/'
    dataloader = GraphDataLoader(root, data_dir='/home/thomas/graph-diffusion-project/graphs_h5', metals=['Cu'])
    print(dataloader)