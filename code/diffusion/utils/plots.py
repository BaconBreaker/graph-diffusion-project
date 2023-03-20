import matplotlib
from matplotlib import pyplot as plt
import os
import networkx as nx
import imageio


def save_graph(batch_dict, t, run_name, post_process):
    matplotlib.use("Agg")

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


def save_gif(run_name, total_t, batch_size, fps=3):
    frames = []
    folder_path = os.path.join("plots", run_name)
    for n in range(batch_size):
        for t in reversed(range(1, total_t + 1)):
            folder_path = os.path.join("plots", run_name)
            file_path = os.path.join(folder_path, f"graph_{n}_timestep_{t}.png")
            image = imageio.v2.imread(file_path)
            frames.append(image)

        gif_path = os.path.join(folder_path, f"graph_{n}.gif")
        imageio.mimsave(gif_path, frames, fps=fps)
