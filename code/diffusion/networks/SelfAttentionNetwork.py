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
    upper_index = torch.triu_indices(adj_matrix.shape[0], adj_matrix.shape[1])
    # Remove the diagonal
    upper_index = upper_index[:,upper_index[0] != upper_index[1]]
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
    edge_seq = batch_dict['edge_sequence']
    pad_mask = batch_dict['pad_mask']
    r = batch_dict['r']
    pdf = batch_dict['pdf']
    
    adj_matrix = torch.zeros(edge_seq.shape[0], pad_mask.shape[1], pad_mask.shape[1])
    upper_index = torch.triu_indices(adj_matrix.shape[1], adj_matrix.shape[2])
    upper_index = upper_index[:,upper_index[0] != upper_index[1]]
    adj_matrix[:, upper_index[0], upper_index[1]] = edge_seq
    adj_matrix[:, upper_index[1], upper_index[0]] = edge_seq
    
    atom_species = batch_dict['node_features'][:, :, 0]

    return adj_matrix, atom_species, r, pdf, pad_mask


class SelfAttentionNetwork(nn.Module):
    def __init__(self, args):
        """
        Self attention network.
        """
        super().__init__()
        self.device = args.device
        self.time_dim = args.time_dim
        self.channels = 1 + args.time_dim
        self.size = args.pad_length

        self.sa1 = EdgeSelfAttention(num_heads=1, channels=self.channels, size=self.size)
        self.sa2 = EdgeSelfAttention(num_heads=1, channels=self.channels, size=self.size)
        self.sa3 = EdgeSelfAttention(num_heads=1, channels=self.channels, size=self.size)
        self.sa4 = EdgeSelfAttention(num_heads=1, channels=self.channels, size=self.size)

        self.out = nn.Linear(self.channels, 1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
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
        x = batch['edge_sequence'].unsqueeze(1)
        mask = batch['pad_mask_sequence']

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        t = t[:, :, None].repeat(1, x.shape[-2], x.shape[-1])
        x = torch.concat([x, t], dim=1)

        # X shape: (batch_size, 1 + time_dim, pad_length * (pad_length - 1) / 2)
        x = x.permute(0, 2, 1) # swap to fit into mha
        x = self.sa1(x, mask)
        x = self.sa2(x, mask)
        x = self.sa3(x, mask)
        x = self.sa4(x, mask)

        x = self.out(x)
        
        batch['edge_sequence'] = x.squeeze(-1)

        return batch
