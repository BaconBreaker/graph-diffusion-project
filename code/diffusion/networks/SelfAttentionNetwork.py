import torch
import torch.nn as nn

from networks.blocks import EdgeSelfAttention


def self_attention_pretransform(sample_dict):
    """
    Transform for self attention network. Just flattens the upper triangle of the adjacency matrix
    into a sequence of edges.
    args:
        sample_dict: dictionary containing tensors:
            adj_matrix: (pad_length, pad_length) tensor containing the adjecency matrix
            node_features: (pad_length, 7) tensor containing the node features
            r: (3000) tensor containing the real space values
            pf: (3000) tensor containing the pair distribution function values
            pad_mask: (pad_length, 1) tensor containing a mask for the padding
    returns:
        sample_dict: dictionary containing tensors:
            edge_sequence: (pad_length * (pad_length - 1) / 2) tensor containing the flattened upper
                triangle of the adjacency matrix
            r: (3000) tensor containing the real space values
            pf: (3000) tensor containing the pair distribution function values
            pad_mask: (pad_length, 1) tensor containing a mask for the padding
            pad_mask_sequence: (pad_length * (pad_length - 1) / 2) tensor containing a mask for the
                flattened upper triangle of the adjacency matrix
    """
    adj_matrix = sample_dict.pop('adj_matrix')
    # Find the upper triangle of the adjacency matrix
    upper_index = torch.triu_indices(adj_matrix.shape[0], adj_matrix.shape[1])
    # Remove the diagonal
    upper_index[:,upper_index[0] != upper_index[1]]
    # Flatten the upper triangle of the adjacency matrix
    edge_sequence = adj_matrix[upper_index[0], upper_index[1]]

    # Make a corresponding mask
    pad_mask = sample_dict['pad_mask']
    n_real = pad_mask.sum() # By summing the mask, we get the number of non-padded nodes
    pad_mask_seq = torch.ones(n_real * (n_real - 1) // 2)
    out_of_bounds = torch.logical_or(upper_index[0] >= n_real-1, upper_index[1] >= n_real-1)
    pad_mask_seq[out_of_bounds] = 0

    sample_dict['edge_sequence'] = edge_sequence
    sample_dict['pad_mask_sequence'] = pad_mask_seq
    
    # Remove unnecessary tensors
    sample_dict.pop('node_features')

    return sample_dict


def self_attention_posttransform(edge_sequence):
    """
    Function to transform to output from a flattened upper triangle of the adjacency matrix to a
    full adjacency matrix.
    args:
        edge_sequence: (pad_length * (pad_length - 1) / 2) tensor containing the flattened upper
            triangle of the adjacency matrix
    """
    # Find the upper triangle of the adjacency matrix
    upper_index = torch.triu_indices(adj_matrix.shape[0], adj_matrix.shape[1])
    # Remove the diagonal
    upper_index[:,upper_index[0] != upper_index[1]]
    # Flatten the upper triangle of the adjacency matrix
    adj_matrix = torch.zeros((edge_sequence.shape[0], edge_sequence.shape[0]))
    adj_matrix[upper_index[0], upper_index[1]] = edge_sequence

    return adj_matrix


class SelfAttentionNetwork(nn.Module):
    def __init__(self, args):
        """
        Self attention network.
        """
        super().__init__()
        self.device = args.device
        self.time_dim = args.time_dim
        self.num_classes = args.num_classes

        self.sa1 = EdgeSelfAttention(num_heads=1, channels=1)
        self.sa2 = EdgeSelfAttention(num_heads=1, channels=1)
        self.sa3 = EdgeSelfAttention(num_heads=1, channels=1)
        self.sa4 = EdgeSelfAttention(num_heads=1, channels=1)

        self.outc = nn.Conv2d(3, 1, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, batch, t, labels=None):
        """
        forward pass of the network
        args:
            batch: batch dictionary containing:
                edge_sequence: (batch_size, pad_length * (pad_length - 1) / 2) tensor containing the
                    flattened upper triangle of the adjacency matrix
                node_features: (batch_size, pad_length, 7) tensor containing the node features
                r: (batch_size, 3000) tensor containing the real space values
                pdf: (batch_size, 3000) tensor containing the pair distribution function values
                pad_mask: (batch_size, pad_length, 1) tensor containing a mask for the padding
            t: (batch_size, 1) tensor containing the time
        """
        x = batch['edge_sequence'].unsqueeze(1)

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        t = t[:, :, None].repeat(1, x.shape[-2], x.shape[-1])

        x = torch.concat([x, t], dim=1)

        x = self.sa1(x)
        x = self.sa2(x)
        x = self.sa3(x)
        x = self.sa4(x)

        x = self.outc(x)
        
        batch['edge_sequence'] = x.squeeze(1)

        return batch
