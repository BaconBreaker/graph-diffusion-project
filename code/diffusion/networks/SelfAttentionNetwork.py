"""
Simple self attention network that operates on adjecency matrix only.

Example run:
python main.py --dataset_path /home/thomas/graph-diffusion-project/graphs_fixed_num_135/ --run_name 123 --max_epochs 1000 --check_val_every_n_epoch 10 --batch_size 4 --tensors_to_diffuse edge_sequence --pad_length 135 --diffusion_timesteps 3 --num_workers 8 --log_every_n_steps 10
"""
import logging

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
            pdf: (3000) tensor containing the pair distribution function values
            pad_mask: (pad_length) tensor containing a mask for the padding
    returns:
        sample_dict: dictionary containing tensors:
            edge_sequence: (pad_length * (pad_length - 1) / 2) tensor containing the flattened upper
                triangle of the adjacency matrix
            r: (3000) tensor containing the real space values
            pdf: (3000) tensor containing the pair distribution function values
            pad_mask: (pad_length) tensor containing a mask for the padding
            pad_mask_sequence: (pad_length * (pad_length - 1) / 2) tensor containing a mask for the
                flattened upper triangle of the adjacency matrix
    """
    adj_matrix = sample_dict.pop('adj_matrix')
    # Find the upper triangle of the adjacency matrix
    upper_index = torch.triu_indices(adj_matrix.shape[0], adj_matrix.shape[1], offset=1)
    # Flatten the upper triangle of the adjacency matrix
    edge_sequence = adj_matrix[upper_index[0], upper_index[1]]

    # Make a corresponding mask
    pad_mask = sample_dict['pad_mask']
    n_real = pad_mask.shape[-1] - pad_mask.sum() # By summing the mask, we get the number of padded nodes
    pad_length = adj_matrix.shape[-1]
    pad_mask_seq = torch.zeros(pad_length * (pad_length - 1) // 2, dtype=torch.bool)
    out_of_bounds = torch.logical_or(upper_index[0] >= n_real-1, upper_index[1] >= n_real-1)
    pad_mask_seq[out_of_bounds] = True

    sample_dict['edge_sequence'] = edge_sequence
    sample_dict['pad_mask_sequence'] = pad_mask_seq

    return sample_dict


def self_attention_posttransform(batch_dict):
    """
    Function to transform to output from a flattened upper triangle of the adjacency matrix to a
    full adjacency matrix. Returns the nessecary tensors to simulate a pdf.
    args:
        batch_dict: dictionary containing tensors:
            edge_sequence: (batch_size, pad_length * (pad_length - 1) / 2) tensor containing the
                flattened upper triangle of the adjacency matrix
            r: (batch_size, 3000) tensor containing the real space values
            pdf: (batch_size, 3000) tensor containing the pair distribution function values
            pad_mask: (batch_size, pad_length) tensor containing a mask for the padding
            pad_mask_sequence: (batch_size, pad_length * (pad_length - 1) / 2) tensor containing a
                mask for the flattened upper triangle of the adjacency matrix
    returns:
        adj_matrix: (batch_size, pad_length, pad_length) tensor containing the adjecency matrix
        r: (batch_size, 3000) tensor containing the real space values
        pdf: (batch_size, 3000) tensor containing the pair distribution function values
        pad_mask: (batch_size, pad_length) tensor containing a mask for the padding
    """
    edge_seq = batch_dict['edge_sequence'].cpu()
    pad_mask = batch_dict['pad_mask'].cpu()
    r = batch_dict['r'].cpu()
    pdf = batch_dict['pdf'].cpu()

    adj_matrix = torch.zeros(edge_seq.shape[0], pad_mask.shape[1], pad_mask.shape[1])
    upper_index = torch.triu_indices(adj_matrix.shape[1], adj_matrix.shape[2])
    upper_index = upper_index[:,upper_index[0] != upper_index[1]]
    adj_matrix[:, upper_index[0], upper_index[1]] = edge_seq
    adj_matrix[:, upper_index[1], upper_index[0]] = edge_seq

    atom_species = batch_dict['node_features'][:, :, 0].cpu()

    return adj_matrix, atom_species, r, pdf, pad_mask


class SelfAttentionNetwork(nn.Module):
    def __init__(self, args):
        """
        Self attention network.
        """
        super().__init__()
        self.time_dim = args.time_dim
        self.conditional = args.conditional
        self.in_channels = 1  # Edge sequence (1)
        self.hidden_channels = 64  # TODO: Make this a parameter
        self.out_channels = 1  # Edge sequence (1)
        self.size = args.pad_length
        self.device = args.device

        self.lin_in = nn.Linear(self.in_channels, self.hidden_channels)
        self.lin_out = nn.Linear(self.hidden_channels, 1)

        layers = []
        for _ in range(10): # TODO: Make this a parameter
            layers.append(EdgeSelfAttention(num_heads=1, channels=self.hidden_channels,
                                            size=self.size, time_dim=self.time_dim,
                                            device=self.device))
        self.layers = nn.ModuleList(layers)

    def pos_encoding(self, t, channels):

        channel_range = torch.arange(0, channels, 2).float().to(self.device)
        inv_freq = 1.0 / (
            10000
            ** (channel_range / channels)
        )
        pos_enc_sin = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_cos = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.zeros(t.shape[0], channels)
        pos_enc[:, 0::2] = pos_enc_sin
        pos_enc[:, 1::2] = pos_enc_cos

        return pos_enc

    def forward(self, batch, t):
        """
        forward pass of the network
        args:
            batch: batch dictionary containing:
                edge_sequence: (batch_size, pad_length * (pad_length - 1) / 2) tensor containing the
                    flattened upper triangle of the adjacency matrix
                r: (batch_size, 3000) tensor containing the real space values
                pdf: (batch_size, 3000) tensor containing the pair distribution function values
                pad_mask: (batch_size, pad_length, 1) tensor containing a mask for the padding
                pad_mask_sequence: (batch_size, pad_length * (pad_length - 1) / 2) tensor containing
                    a mask for the flattened upper triangle of the adjacency matrix
            t: (batch_size, 1) tensor containing the time
        """
        x = batch['edge_sequence'].unsqueeze(-1)
        mask = batch['pad_mask_sequence']

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        x = self.lin_in(x)

        for layer in self.layers:
            x = layer(x, t, mask=mask)

        x = self.lin_out(x)
    
        batch['edge_sequence'] = x.squeeze(-1)

        return batch

