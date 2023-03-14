"""
Equiavariant network for the diffusion process as described in https://arxiv.org/pdf/2203.17003.pdf

Example run: python main.py --model equivariant --dataset_path /home/thomas/graph-diffusion-project/graphs_fixed_num_135/ --run_name 123 --pad_length 135 --tensors_to_diffuse xyz_atom_species --diffusion_timesteps 5 --batch_size 2 --num_workers 8 --check_val_every_n_epoch 10 --max_epochs 100
"""
import torch
import torch.nn as nn
import math

import torch.nn.functional as F


def equivariant_pretransform(sample_dict):
    """
    Extract positions and atom species from node features individually
    """
    node_features = sample_dict['node_features']
    atom_species = node_features[:, 0]
    
    # Save the original atom species
    sample_dict['metal_type'] = atom_species[atom_species != 8.0][0]
    
    # Atom species = -1 if oxygen, 1 if metal
    atom_species[atom_species == 8.0] = -1
    atom_species[atom_species != -1] = 1

    # Coordinates
    xyz = node_features[:, 4:7]
    sample_dict['xyz_atom_species'] = torch.cat((xyz, atom_species[:, None]), dim=1)

    return sample_dict


def equivariant_posttransform(batch_dict):
    xyz_atom_species = batch_dict['xyz_atom_species']
    atom_species_con = xyz_atom_species[:, :, 3]
    xyz = xyz_atom_species[:, :, :3]
    r = batch_dict['r']
    pdf = batch_dict['pdf']
    pad_mask = batch_dict['pad_mask']
    metal_type = batch_dict['metal_type']

    atom_species_dis = torch.zeros_like(atom_species_con, dtype=torch.uint8)
    atom_species_dis[atom_species_con < 0.0] = 8
    atom_species_dis[~torch.isfinite(atom_species_con)] = 8

    for b in range(atom_species_dis.shape[0]):
        atom_species_dis[b][atom_species_con[b] >= 0.0] = metal_type[b].item()

    return xyz, atom_species_dis, r, pdf, pad_mask


class EquivariantNetwork(nn.Module):
    def __init__(self, args):
        """
        Equivariant network.
        """
        super().__init__()
        self.features_in = 12  # Atom species (1) + time (1) + pdf conditioning (10)
        self.features_out = 1  # Atom species (1)
        self.pad_length = args.pad_length
        self.T = args.diffusion_timesteps
        self.hidden_dim = 32  # TODO: make this a parameter

        self.emb_in = nn.Linear(self.features_in, self.hidden_dim)
        self.emb_out = nn.Linear(self.hidden_dim, self.features_out)
        self.x_out = nn.Linear(3, 3)  # To allow some rescaling of the positions
        
        self.pdf_emb = nn.Linear(3000, 10)
        
        self.time_emb = GammaNetwork()

        layers = []
        for _ in range(4):  # TODO: make this a parameter
            layers.append(EGCLayer(self.hidden_dim))
        self.layers = nn.Sequential(*layers)

    def pos_encoding(self, t):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, 2, 2).float() / 2)
        )
        pos_enc_a = torch.sin(t.repeat(1, 1) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, 1) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, batch, t):
        """
        forward pass of the network
        args:
            batch: batch dictionary containing:
                xyz: (batch_size, pad_length, 3) tensor containing the coordinates
                atom_species: (batch_size, pad_length) tensor containing the atom species
                r: (batch_size, 3000) tensor containing the real space values
                pdf: (batch_size, 3000) tensor containing the pair distribution function values
                pad_mask: (batch_size, pad_length) tensor containing a mask for the padding
            t: (batch_size, 1) tensor containing the time
        """
        xh = batch['xyz_atom_species']
        pdf = batch['pdf']

        h = xh[:, :, 3].unsqueeze(-1)
        x = xh[:, :, :3]
        x0 = x.clone()

        # Normalize to 0-1
        t = (t / self.T).unsqueeze(-1)

        # Embedding
        t_emb = self.time_emb(t)
        pdf_emb = self.pdf_emb(pdf)
        t_emb = t_emb.unsqueeze(1).repeat(1, self.pad_length, 1)
        pdf_emb = pdf_emb.unsqueeze(1).repeat(1, self.pad_length, 1)
        h = torch.cat([h, t_emb, pdf_emb], dim=-1)
        h = self.emb_in(h)
        for layer in self.layers:
            x, h = layer(x, h, x0)

        h = self.emb_out(h)
        x = self.x_out(x)
        x = x - x0

        batch['xyz_atom_species'] = torch.cat((x, h), dim=-1)

        return batch


class EGCLayer(nn.Module):
    def __init__(self, n_cat_features):
        """
        EGNN layer as described in appendix B of https://arxiv.org/pdf/2203.17003v2.pdf
        """
        super().__init__()
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.lnh = nn.LayerNorm([n_cat_features])
        self.lnx = nn.LayerNorm([3])

        # Edge operation
        # Input = concat([hi, hj, distance from xi to x2j, distance from xi_0 to xj_0])
        # Output m_ij
        self.edg1 = nn.Linear(n_cat_features * 2 + 2, n_cat_features)
        self.edg2 = nn.Linear(n_cat_features, n_cat_features)

        # Edge inferrence
        # Input = m_ij
        # Output = e_ij
        self.edgi = nn.Linear(n_cat_features, 1)

        # Node feature update
        # Input = concat([hi, aggregated_j m_ij])
        self.node1 = nn.Linear(n_cat_features * 2, n_cat_features)
        self.node2 = nn.Linear(n_cat_features, n_cat_features)
        
        # Node coordinate update
        self.cor1 = nn.Linear(n_cat_features * 2 + 2, n_cat_features)
        self.cor2 = nn.Linear(n_cat_features, n_cat_features)
        self.cor3 = nn.Linear(n_cat_features, 1)

    def edge_operation(self, h1h2, r, r0):
        inp = torch.cat([h1h2, r, r0], dim=-1)
        inp = self.silu(self.edg1(inp))
        inp = self.silu(self.edg2(inp))
        return inp

    def edge_inference(self, m):
        inp = self.sigmoid(self.edgi(m))
        return inp

    def node_update(self, h, m):
        inp = torch.cat([h, m], dim=-1)
        inp = self.silu(self.node1(inp))
        inp = self.node2(inp)
        inp = h + inp
        return inp

    def coordinate_update(self, h1h2, r, r0):
        inp = torch.cat([h1h2, r, r0], dim=-1)
        inp = self.silu(self.cor1(inp))
        inp = self.silu(self.cor2(inp))
        inp = self.cor3(inp)
        return inp

    def forward(self, x, h, x0):
        """
        forward pass of the layer
        args:
            x: (batch_size, pad_length, 3) tensor containing the coordinates
            h: (batch_size, pad_length, n_cat_features) tensor containing the node features
            x0: (batch_size, pad_length, 3) tensor containing the initial coordinates
        """
        org_distances = torch.norm(x0[:, :, None, :] - x0[:, None, :, :], dim=-1).unsqueeze(-1)      
        differences = (x[:, :, None, :] - x[:, None, :, :])
        distances = torch.norm(differences, dim=-1).unsqueeze(-1)
        distances_squarred = distances ** 2
        
        # Layer norm on node features
        h = self.lnh(h)
        x = self.lnx(x)

        # Cartesian product of all nodes
        idx_pairs_cart = torch.cartesian_prod(torch.arange(h.shape[1]), torch.arange(h.shape[1]))
        h_cart = h[:, idx_pairs_cart].view(h.shape[0], h.shape[1], h.shape[1], -1)

        # Edge operation
        m_matrix = self.edge_operation(h_cart, distances_squarred, org_distances)

        # Edge inference
        e_matrix = self.edge_inference(m_matrix)
        diag = torch.eye(e_matrix.shape[1], e_matrix.shape[2]).unsqueeze(0).unsqueeze(-1)
        e_matrix =  e_matrix * (1 - diag)  # Remove self-edges

        # Node update
        h_next = self.node_update(h, torch.sum(e_matrix * m_matrix, dim=-2))

        # Coordinate update
        cor_weight = self.coordinate_update(h_cart, distances_squarred, org_distances)
        cor_shift = differences / (distances + 1)
        cor_update = torch.sum(cor_weight * cor_shift, dim=-2)
        x_next = x + cor_update

        return x_next, h_next


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma

class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)), requires_grad=True)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = F.softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)
