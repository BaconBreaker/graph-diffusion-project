import torch_geometric.utils
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(
        E.shape[1], dtype=torch.bool
    ).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(
        edge_index, edge_attr
    )
    # TODO: carefully check if setting node_mask as a bool 
    # TODO: breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index,
                     batch=batch,
                     edge_attr=edge_attr,
                     max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = n_nodes - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

        self.input_dims = {'X': example_batch['x'].size(1),
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + 1}      # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {'X': example_batch['x'].size(1),
                            'E': example_batch['edge_attr'].size(1),
                            'y': 0}


class MoleculeDatasetInfo(AbstractDatasetInfos):
    def __init__(self, n_nodes, n_node_types, n_edge_types):
        self.n_nodes = n_nodes
        self.n_node_types = n_node_types
        self.n_edge_types = n_edge_types
        n_nodes_tensor = torch.arange(n_nodes)
        self.complete_infos(n_nodes, node_types)
