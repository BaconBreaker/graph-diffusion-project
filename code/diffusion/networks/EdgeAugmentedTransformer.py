"""
Edge aumented graph transformer as described in https://arxiv.org/pdf/2108.03348.pdf

Example run:
python main.py --dataset_path /home/thomas/graph-diffusion-project/graphs_fixed_num_135/ --run_name 123 --max_epochs 20 --check_val_every_n_epoch 10 --batch_size 4 --tensors_to_diffuse adj_matrix atom_species --pad_length 135 --diffusion_timesteps 1000 --num_workers 8 --log_every_n_steps 10 --model eagt --disable_carbon_tracker --metrics mse psnr snr

@Author Thomas Christensen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def EAGTPretransform(sample_dict):
    """
    Pretransforms a sample dict to be compatible with the EAGT model.
    """
    node_features = sample_dict['node_features']
    atom_species = node_features[:, 0]
    adj_matrix = sample_dict['adj_matrix']

    # Save the original atom species
    sample_dict['metal_type'] = atom_species[atom_species != 8.0][0]

    # Atom species = -1 if oxygen, 1 if metal
    atom_species[atom_species == 8.0] = -1
    atom_species[atom_species != -1] = 1
    
    # normalize the adjacency matrix by normalization constant 30
    adj_matrix = adj_matrix / 30.0

    sample_dict['atom_species'] = atom_species
    sample_dict['adj_matrix'] = adj_matrix

    return sample_dict


def EAGTPosttransform(batch_dict):
    """
    Posttransforms a sample dict from EAGT format to the original format.
    """
    adj_matrix = batch_dict['adj_matrix']
    atom_species = batch_dict['atom_species']
    r = batch_dict['r']
    pdf = batch_dict['pdf']
    pad_mask = batch_dict['pad_mask']
    metal_type = batch_dict['metal_type']

    # Convert atom species back to original format
    atom_species_dis = torch.zeros_like(atom_species, dtype=torch.uint8)
    atom_species_dis[atom_species < 0.0] = 8
    atom_species_dis[~torch.isfinite(atom_species)] = 8
    
    # Set metal type
    for b in range(atom_species_dis.shape[0]):
        atom_species_dis[b][atom_species[b] >= 0.0] = metal_type[b].item()
    
    # Convert adjacency matrix back to original format
    adj_matrix = adj_matrix * 30.0

    return adj_matrix, atom_species_dis, r, pdf, pad_mask


class EdgeAugmentedGraphTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.features_in = 11  # Atom species (1) + pdf conditioning (10)
        self.features_out = 1  # Atom species (1)
        self.features_in_edge = 11  # Edge features (1) + pdf conditioning (10)
        self.features_out_edge = 1  # Edge features (1)
        self.pad_length = args.pad_length
        self.hidden_dim = 32  # TODO: Make this a parameter
        self.n_layers = 4  # TODO: Make this a parameter

        self.emb_in = nn.Sequential(
            nn.Linear(self.features_in, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )

        self.emb_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.features_out)
        )

        self.edge_in = nn.Sequential(
            nn.Linear(self.features_in_edge, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )

        self.edge_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.features_out_edge)
        )

        self.pdf_emb = nn.Linear(3000, 10)

        layers = []
        for _ in range(self.n_layers):
            layers.append(EAGTLayer(self.hidden_dim, self.hidden_dim, self.pad_length))
        self.layers = nn.Sequential(*layers)

    def forward(self, batch_dict, t):
        nodes = batch_dict['atom_species']
        edges = batch_dict['adj_matrix']
        pad_mask = batch_dict['pad_mask']
        pdf = batch_dict['pdf']

        pdf_emb = self.pdf_emb(pdf)
        pdf_emb_h = pdf_emb.unsqueeze(1).repeat(1, self.pad_length, 1)
        pdf_emb_e = pdf_emb.unsqueeze(1).unsqueeze(1).repeat(1, self.pad_length, self.pad_length, 1)

        h_in = torch.cat([nodes.unsqueeze(-1), pdf_emb_h], dim=-1)
        h_in = self.emb_in(h_in)
        e_in = torch.cat([edges.unsqueeze(-1), pdf_emb_e], dim=-1)
        e_in = self.edge_in(e_in)

        for layer in self.layers:
            h_in, e_in = layer(h_in, e_in)
        h_out = self.emb_out(h_in)
        e_out = self.edge_out(e_in)

        batch_dict['adj_matrix'] = e_out.squeeze()
        batch_dict['atom_species'] = h_out.squeeze()

        return batch_dict


class EAGTLayer(nn.Module):
    def __init__(self, node_width, edge_width, pad_length, heads=8, clip_min=-5, clip_max=5):
        super().__init__()
        self.node_width = node_width
        self.edge_width = edge_width
        # self.dot_dim = node_width // heads
        self.heads = heads
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Dropout used throughout the model
        self.dropout = nn.Dropout(0.1)

        # Input layernorm layers
        self.ln_e = nn.LayerNorm(edge_width)
        self.ln_n = nn.LayerNorm(node_width)

        self.attention = EdgeAugmentedSelfAttention(node_width, edge_width, heads, clip_min, clip_max)

        # Linear layer for nodes and edges before residual connection
        self.lin_n = nn.Linear(node_width, node_width)
        self.lin_e = nn.Linear(heads, edge_width)

        # Fully connected edge layer
        self.ffn_e = nn.Sequential(
            nn.LayerNorm(edge_width),
            nn.Linear(edge_width, edge_width * 2),
            nn.ELU(),
            nn.Linear(edge_width * 2, edge_width)
        )

        # Fully connected node layer
        self.ffn_n = nn.Sequential(
            nn.LayerNorm(node_width),
            nn.Linear(node_width, node_width * 2),
            nn.ELU(),
            nn.Linear(node_width * 2, node_width)
        )

        # Final layer norms
        self.ln_e_final = nn.LayerNorm(edge_width)
        self.ln_n_final = nn.LayerNorm(node_width)
    
    def forward(self, nodes, edges):
        """
        Args:
            nodes: [B, N, node_width]
            edges: [B, N, N, edge_width]
        """
        # LayerNorm
        e = self.ln_e(edges)
        n = self.ln_n(nodes)

        # Get attention
        attn, H = self.attention(n, e)

        # Get edge features
        e_hat = self.lin_e(H)
        e_hat = self.dropout(e_hat)
        e_hat = e_hat + edges

        # Get node features
        n_hat = self.lin_n(attn)
        n_hat = self.dropout(n_hat)
        n_hat = n_hat + nodes

        # Last overfull linear layers
        e_hat = e_hat + self.ffn_e(e_hat)
        n_hat = n_hat + self.ffn_n(n_hat)

        # Last layer norms
        e_hat = self.ln_e_final(e_hat)
        n_hat = self.ln_n_final(n_hat)
        
        # Set diagonal to zero and symmetrize the adjacency matrix
        triu_indices = torch.triu_indices(e_hat.shape[1], e_hat.shape[1], offset=1)
        e_hat[:, triu_indices[1], triu_indices[0]] = e_hat[:, triu_indices[0], triu_indices[1]]
        e_hat =  e_hat * (1 - torch.eye(e_hat.shape[1], e_hat.shape[1]).reshape(1, e_hat.shape[1], e_hat.shape[1], 1))

        return n_hat, e_hat


class EdgeAugmentedSelfAttention(nn.Module):
    def __init__(self, node_width, edge_width, heads=8, clip_min=-5, clip_max=5):
        super().__init__()
        self.node_width = node_width
        self.edge_width = edge_width
        self.dot_dim = node_width // heads
        self.heads = heads
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        self.q = nn.Linear(node_width, node_width)
        self.k = nn.Linear(node_width, node_width)
        self.v = nn.Linear(node_width, node_width)

        self.g = nn.Linear(edge_width, heads)
        self.e = nn.Linear(edge_width, heads)

        self.dropout = nn.Dropout(0.1)

    def forward(self, nodes, edges):
        """
        Args:
            nodes: [B, N, node_width]
            edges: [B, N, N, edge_width]
        """
        # Get query, key and value
        q = self.q(nodes).view(nodes.shape[0], nodes.shape[1], self.dot_dim, self.heads)
        k = self.k(nodes).view(nodes.shape[0], nodes.shape[1], self.dot_dim, self.heads)
        v = self.v(nodes).view(nodes.shape[0], nodes.shape[1], self.dot_dim, self.heads)

        # Get attention scores
        H = torch.einsum('bldh, bmdh -> blmh', q, k)
        H = H / (self.dot_dim ** 0.5)

        # Clip attention scores
        H = torch.clamp(H, self.clip_min, self.clip_max)

        # Get edge features
        g = self.g(edges)
        e = self.e(edges)

        # Add bias term to attention scores
        H = H + e

        # Gate the attention scores
        A = torch.sigmoid(g) * F.softmax(H, dim=2)

        # Apply dropout
        A = self.dropout(A)
        
        # Get attention values
        attn = torch.einsum('blmh, bmdh -> bldh', A, v)
        attn = attn.reshape(nodes.shape[0], nodes.shape[1], self.node_width)

        return attn, H
