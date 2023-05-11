"""
Equiavariant network for the diffusion process as described in https://arxiv.org/pdf/2203.17003.pdf

Example run: python main.py --model equivariant --dataset_path /home/thomas/graph-diffusion-project/graphs_fixed_num_135/ --run_name 123 --pad_length 135 --tensors_to_diffuse xyz_atom_species --diffusion_timesteps 5 --batch_size 2 --num_workers 8 --check_val_every_n_epoch 10 --max_epochs 100
"""
import torch
import torch.nn as nn
import math
import random

import torch.nn.functional as F


def equivariant_pretransform(sample_dict):
    """
    Extract positions and atom species from node features individually
    """
    node_features = sample_dict['node_features']

    # Coordinates
    xyz = node_features[:, 4:7]
    sample_dict['xyz'] = xyz

    return sample_dict


def equivariant_posttransform(batch_dict):
    xyz = batch_dict['xyz']
    r = batch_dict['r']
    pdf = batch_dict['pdf']
    pad_mask = batch_dict['pad_mask']

    atom_species = torch.ones(xyz.shape[0], xyz.shape[1], 1).to(xyz.device) * 6  # Carbon

    return xyz, atom_species, r, pdf, pad_mask


class EquivariantNetwork(nn.Module):
    def __init__(self, args):
        """
        Equivariant network.
        """
        super().__init__()
        self.features_out = 1  # Atom species (1)
        self.pad_length = args.pad_length
        self.T = args.diffusion_timesteps
        self.hidden_dim = args.equiv_hidden_dim
        self.pdf_hidden_dim = args.equiv_pdf_hidden_dim
        self.n_layers = args.equiv_n_layers
        self.conditional = args.conditional
        self._edges_dict = {}

        self.emb_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.features_out)
        )

        self.pdf_emb = nn.Sequential(
            nn.Linear(3000, self.pdf_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.pdf_hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.time_emb = nn.Sequential(
            GammaNetwork(),
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.h_emb = nn.Sequential(
            nn.Linear(1, 10),
            nn.GELU(),
            nn.Linear(10, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Layer that takes pdf_emb and t_emb and combines them
        # into scale and shift values
        self.film_emb = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        layers = []
        for _ in range(self.n_layers):
            layers.append(EQLayer(self.hidden_dim, self.n_layers))
        self.layers = nn.Sequential(*layers)

    def pos_encoding(self, t):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 2, 2).float() / 2))
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
        x = batch['xyz']
        pdf = batch['pdf']
        pad_mask = batch['pad_mask']
        pad_mask = pad_mask.to(x.device)
        pdf = pdf.to(x.device)

        # Flip padding values so 1 means no padding and 0 means padding
        # Also unsqueeze to (batch_size, pad_length, 1) and convert to float
        pad_mask = (~pad_mask).unsqueeze(-1).float()
        x0 = x.clone()

        # Normalize to 0-1 and unsqueeze to (batch_size, 1)
        t = (t / self.T).unsqueeze(-1)

        # Subtact mean to center molecule at origin
        N = pad_mask.sum(1)
        mean = (torch.sum(x, dim=1) / N).unsqueeze(1)
        x = (x - mean) * pad_mask

        # Embedding in
        t_emb = self.time_emb(t)
        t_emb = t_emb.unsqueeze(1).repeat(1, self.pad_length, 1)
        h_emb = self.h_emb(torch.ones(x.shape[0], x.shape[1], 1).to(x.device))

        # Embedding for pdf if we want to condition on it
        if self.conditional:
            pdf_emb = self.pdf_emb(pdf)
            pdf_emb = pdf_emb.unsqueeze(1).repeat(1, self.pad_length, 1)
        else:
            pdf_emb = torch.zeros_like(t_emb)

        # Scale shift conditioning embedding
        scaleshift = self.film_emb(torch.cat((pdf_emb, t_emb), dim=-1))
        scale, shift = scaleshift.chunk(2, dim=-1)

        # dropout the conditioning with 10% probability
        if self.training and random.random() < 0.1:
            h = h_emb
        else:
            h = scale * h_emb + shift

        # Calculate edge indicies
        edges = self.get_adj_matrix(self.pad_length, x.size(0))

        # Reshape x and h
        bs = x.shape[0]
        x = x.reshape(bs * self.pad_length, -1)
        h = h.reshape(bs * self.pad_length, -1)

        # Calculate original distances
        distances, _ = coord2diff(x, edges)

        # Loop over layers
        for layer in self.layers:
            h, x = layer(h, x, edges, distances)

        # Reshape x and h
        x = x.reshape(bs, self.pad_length, -1)
        h = h.reshape(bs, self.pad_length, -1)

        # Embedding out
        h = self.emb_out(h)

        # Rescale to center of gravity
        vel = x - x0

        # Re-pad values before returning
        if pad_mask is not None:
            vel = vel * pad_mask

        # Remove mean
        N = pad_mask.sum(1)
        mean = (torch.sum(vel, dim=1) / N).unsqueeze(1)
        vel = (vel - mean) * pad_mask

        batch['xyz'] = vel
        return batch

    def pad_noise(self, noise, batch_dict):
        """
        Pads the noise to the correct length
        Sets the noise for the padding to 0
        args:
            noise: (batch_size, pad_length, 4) tensor containing the noise
            batch_dict: batch dictionary containing:
                pad_mask: (batch_size, pad_length) tensor containing a mask for the padding
        returns:
            noise: (batch_size, pad_length, 4) tensor containing the padded noise
        """
        pad_mask = batch_dict['pad_mask']
        pad_mask = pad_mask.unsqueeze(-1).repeat(1, 1, 3)  # (batch_size, pad_length, 3)
        noise[pad_mask] = 0.0

        return noise

    def get_adj_matrix(self, n_nodes, batch_size):
        """
        Calculates edge indicies for a batch of molecules using dynamic programming
        """
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows), torch.LongTensor(cols)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size)


class EQLayer(nn.Module):
    def __init__(self, hidden_dim, n_total_layers, n_layers=1):
        super(EQLayer, self).__init__()

        self.gcls = nn.ModuleList([GCL(hidden_dim) for _ in range(n_layers)])
        self.coord_update = EQUpdate(hidden_dim, n_total_layers)

    def forward(self, h, x, edges, distances_org):  # TODO: support masking
        distances, coord_diff = coord2diff(x, edges)
        for gcl in self.gcls:
            h = gcl(h, edges, distances)
        x = self.coord_update(h, x, edges, coord_diff, distances, distances_org)

        return h, x


class GCL(nn.Module):
    def __init__(self, hidden_dim):
        super(GCL, self).__init__()
        self.hidden_dim = hidden_dim

        self.edg1 = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.edg2 = nn.Linear(hidden_dim, hidden_dim)

        self.edgi = nn.Linear(hidden_dim, 1)

        self.node1 = nn.Linear(hidden_dim + 1, hidden_dim)
        self.node2 = nn.Linear(hidden_dim, hidden_dim)

        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def edge_operation(self, h1, h2, r):
        inp = torch.cat([h1, h2, r], dim=1)
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

    def forward(self, h, edges, distances):
        row, col = edges
        edge_feat = self.edge_operation(h[row], h[col], distances)
        edge_feat = self.edge_inference(edge_feat) * edge_feat
        agg = unsorted_segment_sum(distances, row, h.shape[0])
        h = self.node_update(h, agg)

        return h


class EQUpdate(nn.Module):
    def __init__(self, hidden_dim, n_total_layers):
        super(EQUpdate, self).__init__()
        self.hidden_dim = hidden_dim
        # coord range is manually set to 12 here
        self.coord_range = 12 / n_total_layers

        self.cor1 = nn.Linear(hidden_dim * 2 + 2, hidden_dim)
        self.cor2 = nn.Linear(hidden_dim, hidden_dim)
        self.cor3 = nn.Linear(hidden_dim, 1, bias=False)

        self.silu = nn.SiLU()

    def coordinate_update(self, h1, h2, r, r0):
        inp = torch.cat([h1, h2, r, r0], dim=1)
        inp = self.silu(self.cor1(inp))
        inp = self.silu(self.cor2(inp))
        inp = self.cor3(inp)
        return inp

    def forward(self, h, x, edges, coord_diff, distances, distance_org):
        row, col = edges
        # trans = coord_diff * self.coordinate_update(h[row], h[col], distances, distance_org)
        trans = coord_diff * F.tanh(self.coordinate_update(h[row], h[col], distances, distance_org)) * self.coord_range
        agg = unsorted_segment_sum(trans, row, x.size(0))

        x = x + agg
        return x


# Copied from the their github
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


# Copied from the their github
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


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    segment_ids = segment_ids.to(data.device)
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0, device=data.device)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    result = result / 100  # Normalization factor ???

    return result
