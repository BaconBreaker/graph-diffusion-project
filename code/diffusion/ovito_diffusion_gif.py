import logging

from main import parse_args, check_args
from utils.get_config import get_model, get_diffusion
from utils.data import load_structure
from scipy.spatial.distance import pdist
import torch
from tqdm.auto import tqdm
from utils.pdf import multidimensional_scaling
from utils.plots import xyz_to_str, save_to_csv
from mendeleev import element
import numpy as np
from sklearn.decomposition import PCA

# Select the single sample beforehand :)

# Size 135
#graphp = "/home/thomas/graph-diffusion-project/graphs_h5/graph_NaCl_KO_r8.h5"

# Size 511
graphp = "/home/thomas/graph-diffusion-project/graphs_h5/graph_Wurtzite_BO_r8.h5"

def generate_gif_adj_matrix(args):
    diffusion_model = get_diffusion(args)
    edge_features, edge_indices, node_features, r, pdf = load_structure(graphp)
    n_nodes = node_features.shape[0]

    # Calculate the adj_matrix
    xyz = node_features[:, 4:7]
    adj_matrix = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    tri_cor = torch.triu_indices(n_nodes, n_nodes, offset=1)
    distances = torch.tensor(pdist(xyz, metric='euclidean'), dtype=torch.float32)
    adj_matrix[tri_cor[0], tri_cor[1]] = distances
    adj_matrix[tri_cor[1], tri_cor[0]] = distances 

    # Used later to rescale the adj_matrix
    t_min = distances.min().item()
    t_max = distances.max().item()

    # collect into dict to pass to diffusion model
    ex_batch = {
            'adj_matrix': adj_matrix.unsqueeze(0),
            'node_features': node_features.unsqueeze(0),
            'r': r.unsqueeze(0),
            'pdf': pdf.unsqueeze(0)}

    if args.fix_noise:
        fixed_noises = [diffusion_model.sample_from_noise_fn(
            ex_batch[tensor].shape) for tensor in diffusion_model.tensors_to_diffuse]
    else:
        fixed_noises = None

    pbar = tqdm(range(0, args.diffusion_timesteps, args.t_skips))
    atom_species = node_features[:, 0]
    
    # Pre-compute the atom species as strings to save time
    atom_species = np.array([element(int(atom)).symbol for atom in atom_species], dtype=object).reshape(-1, 1)

    total_string = xyz_to_str(xyz, atom_species)
    for t in pbar:
        diffused_batch, _ = diffusion_model.diffuse(ex_batch, t, fixed_noises)
        adj_matrix = diffused_batch['adj_matrix'][0]

        #Rescaling acording to https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        adj_logits = adj_matrix.triu(diagonal=1)[adj_matrix.triu(diagonal=1) != 0]
        r_min = adj_logits.min().item()
        r_max = adj_logits.max().item()
        adj_matrix = ((adj_matrix - r_min) / (r_max - r_min)) * (t_max - t_min) + t_min
        
        # Convert to point cloud and rotate using PCA
        xyz = multidimensional_scaling(adj_matrix)
        xyz = PCA(n_components=3).fit_transform(xyz)
        
        # Save to string
        s = xyz_to_str(xyz, atom_species)
        total_string += s

    save_to_csv("/home/thomas/adj.xyz", total_string)

def generate_gif_xyz(args):
    diffusion_model = get_diffusion(args)
    edge_features, edge_indices, node_features, r, pdf = load_structure(graphp)
    xyz = node_features[:, 4:7]
    atom_species = node_features[:, 0]
    atom_species = np.array([element(int(atom)).symbol if atom else atom for atom in atom_species]).reshape(-1, 1)
    
    ex_batch = {
        'xyz': xyz.unsqueeze(0),
        'node_features': node_features.unsqueeze(0),
        'r': r.unsqueeze(0),
        'pdf': pdf.unsqueeze(0)}

    if args.fix_noise:
        fixed_noises = [diffusion_model.sample_from_noise_fn(
            ex_batch[tensor].shape) for tensor in diffusion_model.tensors_to_diffuse]
    else:
        fixed_noises = None

    pbar = tqdm(range(0, args.diffusion_timesteps, args.t_skips))
    total_string = xyz_to_str(xyz, atom_species)

    for t in pbar:
        diffused_batch, _ = diffusion_model.diffuse(ex_batch, t, fixed_noises)
        xyz = diffused_batch['xyz'][0]
        s = xyz_to_str(xyz, atom_species)
        total_string += s
    
    save_to_csv("/home/thomas/xyz.xyz", total_string)

def main():
    args = parse_args()
    check_args(args)
    if args.method == "adj_matrix":
        generate_gif_adj_matrix(args)   
    else:
        generate_gif_xyz(args)

if __name__ == "__main__":
    main()