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
from copy import copy
import numpy.matlib
from matplotlib.colors import LogNorm


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


def make_histograms(positions, save_path):
    """
    Funciton to make histograms of positions and distances between atoms
    args:
        positions: torch.tensor of shape (frames, n_atoms, 3)
        save_path: str, path to save the histograms
    """
    make_histogram_single(positions, save_path + "positions_hist", plot_positions=True)
    make_histogram_single(positions, save_path + "distances_hist", plot_positions=False)


def make_histogram_single(positions, save_path, plot_positions=True):
    """
    Function to make a histogram of positions or distances between atoms
    args:
        positions: torch.tensor of shape (frames, n_atoms, 3)
        save_path: str, path to save the histogram
        plot_positions: bool, if True plot positions, else plot distances
    """
    fig, axes = plt.subplots(nrows=3, figsize=(6, 8), layout='constrained')
    if plot_positions:
        fig.suptitle("position value histograms over time")
        axes[1].set_ylabel("position values")
        axes[0].set_ylabel("position values")
        axes[2].set_ylabel("position values")
        x = np.arange(positions.shape[0])
        Y = positions.reshape(positions.shape[0], -1).T
        Y = np.nan_to_num(Y)
    else:
        fig.suptitle("distance histograms over time")
        axes[1].set_ylabel("distances")
        axes[0].set_ylabel("distances")
        axes[2].set_ylabel("distances")
        x = np.arange(positions.shape[0])
        Y = torch.cdist(positions, positions, p=2).reshape(positions.shape[0], -1).T
        Y = np.nan_to_num(Y)

    axes[0].plot(x, Y.T, color="C0", alpha=0.1)
    axes[0].set_title("Line plot with alpha")
    axes[0].set_xlabel("frames")

    num_fine = positions.shape[0]
    num_series = Y.shape[0]
    x_fine = np.linspace(x.min(), x.max(), num_fine)
    y_fine = np.empty((num_series, num_fine), dtype=float)
    for i in range(num_series):
        y_fine[i, :] = np.interp(x_fine, x, Y[i, :])
    y_fine = y_fine.flatten()
    x_fine = np.matlib.repmat(x_fine, num_series, 1).flatten()

    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))
    h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[400, 100])
    pcm = axes[1].pcolormesh(xedges, yedges, h.T, cmap=cmap,
                             norm=LogNorm(vmax=1.5e2), rasterized=True)
    # fig.colorbar(pcm, ax=axes[1], label="# points", pad=0)
    axes[1].set_title("2d histogram and log color scale")
    axes[1].set_xlabel("frames")

    # Same data but on linear color scale
    pcm = axes[2].pcolormesh(xedges, yedges, h.T, cmap=cmap,
                             vmax=1.5e2, rasterized=True)
    fig.colorbar(pcm, ax=axes[2], label="# points", pad=0)
    axes[2].set_title("2d histogram and linear color scale")
    axes[2].set_xlabel("frames")

    plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
