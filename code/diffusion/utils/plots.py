import matplotlib
from matplotlib import pyplot as plt
import os
import networkx as nx
import imageio
import numpy as np
from mendeleev import element


def save_graph(batch_dict, t, run_name, post_process):
    matplotlib.use("Agg")

    if t % 10 != 0:
        return

    # After post-processing all batches have the same structure.
    post_batch = post_process(batch_dict)

    adjacency_matrices = post_batch[0].cpu().detach().numpy()

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
        atom_species = np.array(["C"]*n_atoms).reshape(-1, 1)

    # if atom_species is number, convert to atomic symbol
    if np.issubdtype(atom_species.dtype, np.number):
        atom_species = np.array(
            [element(int(atom)).symbol for atom in atom_species]).reshape(-1, 1)

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

def save_to_csv(file_name, s):
    with open(file_name, "w") as f:
        f.write(s)