# Atomic structure graphs

This particular dataset contains 1728 spherical structure graphs (nanoparticles), ranging from 5 to 8 Ångstrom (0.5-0.8 nm) in size. (We can make them bigger if you like)
The structure are mono-metallic metal-oxide structure, meaning they all contain oxigen "ligands" and some metal species (but only one species per structure).

The graphs are stored in ´.h5´ files, that can be opened using the h5py package (pip install h5py). 
Here I am using pytorch geometric as the graph dataframe:

```
  import h5py
  import torch
  from pytorch_geometric import Data

  with h5py.File('some_path.h5', 'r') as file:
    edge_features = torch.tensor(file['edge_attributes'][:], dtype=torch.float32) # Edge attributes
    edge_indices = torch.tensor(file['edge_indices'][:], dtype=torch.long) # Edge (sparse) adjecency matrix
    node_features = torch.tensor(file['node_attributes'][:], dtype=torch.float32) # Node attributes

    r = file['r'][...] # Real space (x-axis)
    pdf = file['pdf'][...] # G(r) (y-axis)

    # Here you can do some normalisation of the node features and perhaps pick out which you want to include.

    graph = Data(x = node_attributes, y = pdf, edge_attr = edge_attributes, edge_index = edge_indices)
```

The node attributes are "atomic number", "atomic radius", "atomic density", "electron affinity", "x-coordinate", "y-coordinate" and "z-coordinate", but you can choose to use any or non of these if you like.
