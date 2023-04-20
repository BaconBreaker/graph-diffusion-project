import torch


def digress_pretransform(sample_dict):
    old_node_features = sample_dict["node_features"]
    adj_matrix = sample_dict["adj_matrix"]

    # For every entry in adj_matrix of shape (b, n, n), if the element is 
    # between i and i+1 it goes into the i-th bin. Goes from 0 to 29, so 30 bins.
    # This goes into the edge_features tensor with shape (b, n, n, 30)
    edge_features = torch.zeros((adj_matrix.shape[0],
                                 adj_matrix.shape[1],
                                 adj_matrix.shape[2],
                                 30))
    for i in range(30):
        edge_features[:, :, :, i] = torch.logical_and(adj_matrix >= i,
                                                      adj_matrix < i+1).float()

    # old_node_features has shape (b, n, 7). node_features has shape (b, n, 2)
    # initialized as zeros. if old_node_features[:, i, 0] == 8, then
    # node_features[:, i, 0] = 1, else node_features[:, i, 1] = 1.
    node_features = torch.zeros((old_node_features.shape[0],
                                 old_node_features.shape[1],
                                 2))
    node_features[:, :, 0] = torch.eq(old_node_features[:, :, 0], 8).float()
    node_features[:, :, 1] = torch.ne(old_node_features[:, :, 0], 1).float()

    sample_dict["edge_features"] = edge_features
    sample_dict["node_features"] = node_features

    return sample_dict
