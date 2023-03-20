"""
Edge aumented graph transformer as described in https://arxiv.org/pdf/2108.03348.pdf

Example run: ...

@Author Thomas Christensen
"""
import torch
import torch.nn as nn
import torch.functional as F


def EAGTPretransform(sample_dict):
    """
    Pretransforms a sample dict to be compatible with the EAGT model.
    """
    node_features = sample_dict['node_features']
    atom_species = node_features[:, 0]
    sample_dict['atom_species'] = atom_species
    
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
    
    return adj_matrix, atom_species, r, pdf, pad_mask


class EdgeAugmentedGraphTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.features_in = 12  # Atom species (1) + time (1) + pdf conditioning (10)
        self.features_out = 1  # Atom species (1)
        self.pad_length = args.pad_length
        self.hidden_dim = 64
        self.n_layers = 4

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
        
        self.pdf_emb = nn.Linear(3000, 10)

        layers = []
        for _ in range(self.n_layers):
            layers.append(EAGTLayer(self.hidden_dim, self.hidden_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, batch_dict, t):
        nodes = batch_dict['atom_species']
        edges = batch_dict['adj_matrix']
        pad_mask = batch_dict['pad_mask']
        pdf = batch_dict['pdf']

        pdf_emb = self.pdf_emb(pdf)


class EAGTLayer(nn.Module):
    def __init__(self, node_width, edge_width, heads=8, clip_min=-5, clip_max=5):
        super().__init__()
        self.node_width = node_width
        self.edge_width = edge_width
        self.dot_dim = node_width // heads
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
        self.lin_e = nn.Linear(self.dot_dim, edge_width)

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

        return n_hat, e_hat


class EdgeAugmentedSelfAttention(nn.Module):
    def __init__(self, node_width, edge_width, heads=8):
        super().__init__()
        self.node_width = node_width
        self.edge_width = edge_width
        self.dot_dim = node_width // heads
        self.heads = heads
        self.clip_min = -5
        self.clip_max = 5
        
        self.q = nn.Linear(node_width, node_width)
        self.k = nn.Linear(node_width, node_width)
        self.v = nn.Linear(node_width, node_width)

        self.g = nn.Linear(edge_width, 1)
        self.e = nn.Linear(edge_width, 1)
        
        self.lin_O_e = nn.Linear(heads, edge_width)
        self.lin_O_h = nn.Linear(node_width, node_width)

        self.dropout = nn.Dropout(0.1)

    def forward(self, nodes, edges):
        """
        Args:
            nodes: [B, N, node_width]
            edges: [B, N, N, edge_width]
        """
        # Get query, key and value
        q = self.q(nodes).view(nodes.shape[0], nodes.shape[1], self.heads, self.dot_dim)
        k = self.k(nodes).view(nodes.shape[0], nodes.shape[1], self.heads, self.dot_dim)
        v = self.v(nodes).view(nodes.shape[0], nodes.shape[1], self.heads, self.dot_dim)

        # Get attention scores
        H = torch.einsum('bijk,bilk->bijl', q, k)
        H = H / (self.dot_dim ** 0.5)

        # Clip attention scores
        H = torch.clamp(H, self.clip_min, self.clip_max)

        # Get edge features
        g = self.g(edges).view(edges.shape[0], edges.shape[1], edges.shape[2], 1)
        e = self.e(edges).view(edges.shape[0], edges.shape[1], edges.shape[2], 1)

        # Add bias term to attention scores
        H = H + e

        # Gate the attention scores
        A = F.sigmoid(g) * F.softmax(H, dim=2)

        # Apply dropout
        A = self.dropout(A)

        # Get output
        attn = torch.einsum('bijl,bilk->bijk', A, v)
        attn = attn.view(nodes.shape[0], nodes.shape[1], nodes.shape[2])

        return attn, H