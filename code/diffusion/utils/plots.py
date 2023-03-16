import matplotlib
from matplotlib import pyplot as plt
import os
import networkx as nx


def save_graph(batch_dict, t, run_name, post_process):
    matplotlib.use("Agg")

    post_batch = post_process(batch_dict)

    adjacency_matrices = post_batch[0].cpu().detach().numpy()

    for n, adjacency_matrix in enumerate(adjacency_matrices):
        folder_path = os.path.join("plots", run_name)
        img_path = os.path.join(folder_path, f"graph_{n}_timestep_{t}.png")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        g = nx.from_numpy_array(adjacency_matrix)
        fig = plt.figure()
        nx.draw(g, ax=fig.add_subplot())
        fig.savefig(img_path)