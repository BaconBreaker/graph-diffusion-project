import matplotlib
from matplotlib import pyplot as plt
import os
import networkx as nx
import imageio
import numpy as np
from mendeleev import element
from sklearn.decomposition import PCA
from utils.pdf import multidimensional_scaling
import torch


def save_graph(batch_dict, t, run_name, post_process):
    matplotlib.use("Agg")

    if t % 10 != 0:
        return

    # After post-processing all batches have the same structure.
    post_batch = post_process(batch_dict)

    adjacency_matrices = post_batch[0].detach().cpu().numpy()

    for n, adjacency_matrix in enumerate(adjacency_matrices):
        folder_path = os.path.join("plots", run_name)
        img_path = os.path.join(folder_path, f"graph_{n}_timestep_{t}.png")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        g = nx.from_numpy_array(adjacency_matrix)
        fig = plt.figure(figsize=(6, 6))
        nx.draw(g, ax=fig.add_subplot())
        plt.title("graph at timestep " + str(t))
        fig.savefig(img_path, transparent=False, facecolor="white")
        plt.close(fig)


def save_gif(run_name: str, total_t: int, batch_size: int,
             fps: int = 3, skips: int = 1):
    if total_t < skips:
        raise ValueError(f"skips ({skips}) is larger than the total t ({total_t})")

    frames = []

    folder_path = os.path.join("plots", run_name)
    for n in range(batch_size):
        for t in reversed(range(skips, total_t + 1, skips)):
            folder_path = os.path.join("plots", run_name)
            file_path = os.path.join(folder_path, f"graph_{n}_timestep_{t}.png")
            image = imageio.v2.imread(file_path)
            frames.append(image)

        gif_path = os.path.join(folder_path, f"graph_{n}.gif")
        imageio.mimsave(gif_path, frames, fps=fps)


def xyz_to_str(xyz, atom_species=None):
    """
    Write a string in format ovito can read
    args:
        xyz: np.array of shape (n_atoms, 3)
        atom_species: np.array of shape (n_atoms, 1)
    returns:
        s: string in ovito format
    """
    n_atoms = xyz.shape[0]
    if atom_species is None:
        atom_species = np.array(["C"] * n_atoms).reshape(-1, 1)
    else:
        atom_species = atom_species.numpy()

    # edge-case when atom_species is a numpy array but has dtype torch.float32
    if atom_species.dtype == torch.float32:
        atom_species = atom_species.astype(np.float32)

    # if atom_species is number, convert to atomic symbol
    # if np.issubdtype(atom_species.dtype, np.number):
    #     atom_species = np.array(
    #         [element(int(atom)).symbol for atom in atom_species]).reshape(-1, 1)

    atom_species = np.array(["C"] * n_atoms).reshape(-1, 1)

    s = ""
    vals = np.concatenate((atom_species, xyz), axis=1)

    # Number of atoms
    s += str(n_atoms) + "\n"

    # Comment line, just keep empty for now
    s += "\n"

    # Coordinates for each atom
    for atom, x, y, z in vals:
        s += f"{atom} {x} {y} {z} \n"

    return s


def adj_to_str(adj_matrix, atom_species=None):
    # Convert to point cloud and rotate using PCA
    xyz = multidimensional_scaling(adj_matrix)
    xyz = PCA(n_components=3).fit_transform(xyz)

    return xyz_to_str(xyz, atom_species)


def save_graph_str_batch(batch_dict, post_process, log_strs):
    # After post-processing all batches have the same structure.
    post_batch = post_process(batch_dict)
    matrices_in = post_batch[0].cpu().detach()
    pad_masks = post_batch[4].cpu().detach()
    atom_species = post_batch[1].cpu().detach()
    strs = []
    for matrix_in, pad_mask, atom_spec in zip(matrices_in, pad_masks, atom_species):
        s = save_graph_str(matrix_in, pad_mask, atom_spec)
        strs.append(s)

    assert len(log_strs) == len(strs)
    log_strs = [a + b for a, b in zip(log_strs, strs)]  # Concatenate strings

    return log_strs


def save_graph_str(matrix_in, pad_mask, atom_species=None):
    # Invert pad mask
    pad_mask = torch.logical_not(pad_mask.bool())
    if matrix_in.shape[1] != 3:  # If matrix_in is adjecency matrix
        adj_matrix = matrix_in[pad_mask][:, pad_mask]
        species = atom_species[pad_mask]
        return adj_to_str(adj_matrix, species)
    else:  # If matrix_in is point cloud
        point_cloud = matrix_in[pad_mask]
        species = atom_species[pad_mask]
        return xyz_to_str(point_cloud, species)


def save_to_file(file_name, s):
    with open(file_name, "w") as f:
        f.write(s)
