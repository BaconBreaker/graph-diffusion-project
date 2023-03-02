import torch
import torchvision
from torch.utils.data import DataLoader

import os
from PIL import Image

import matplotlib.pyplot as plt

from sklearn.manifold import MDS

from diffpy.structure import Structure, Atom
from diffpy.srreal.pdfcalculator import DebyePDFCalculator

from mendeleev import element

from scipy.stats import pearsonr


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1)
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args, subset=None):
    if args.graph_data:
        return get_data_graph(args, subset)
    else:
        return get_data_(args, subset)


def get_data_graph(args, subset=None):
    adjecency_matrix = torch.load(os.path.join(args.dataset_path, "adjecency_tensor_train.pt"))
    node_features = torch.load(os.path.join(args.dataset_path, "node_tensor_train.pt"))
    pdfs = torch.load(os.path.join(args.dataset_path, "pdf_tensor_train.pt"))
    if subset is not None:
        adjecency_matrix = adjecency_matrix[:subset]
        node_features = node_features[:subset]
        pdfs = pdfs[:subset]

    # This is just to make it fit into the image setup.
    adjecency_matrix = adjecency_matrix[None, :, None, :134, :134]
    node_features = node_features[None, :, :134, :]
    pdfs = pdfs[None, ...]
    labels_tmp = torch.ones(adjecency_matrix.shape[0], dtype=torch.long)
    pad_mask = torch.ones(node_features.shape[0:3], dtype=bool)

    dataloader = torch.utils.data.TensorDataset(adjecency_matrix, labels_tmp, node_features, pdfs, pad_mask)

    return dataloader


def get_data_(args, subset=None):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(88),
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    if subset is not None:
        dataset = torch.utils.data.Subset(dataset, range(subset))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def multidimensional_scaling(adj_matrix):
    """
    Converet adjecency matrix to point cloud using multidimensional scaling
    args:
        adj_matrix (np.array): adjecency matrix of size (n, n)
    returns:
        point cloud (np.array): point cloud of size (n, 3)
    """
    mds = MDS(
        n_components=3,
        metric=False,
        max_iter=3000,
        eps=1e-9,
        dissimilarity="precomputed",
        n_jobs=1
    )
    point_cloud = mds.fit_transform(adj_matrix)
    return point_cloud


def calculate_pdf(point_cloud, atom_species):
    """
    Calculate pdf from point cloud
    args:
        point_cloud (np.array): point cloud of size (n, 3)
        atom_species (np.array): atom species of size (n,) containing their atomic weghts
    returns:
        pdf (np.array): array of evaluated points of the pdf function of shape (3000,)
    """
    structure = Structure()
    for atom, xyz in zip(atom_species, point_cloud):
        if not isinstance(atom.item(), int):
            atom = int(atom.item())
        else:
            atom = atom.item()
        structure.append(Atom(element(atom).symbol, xyz))

    structure.B11 = 0.3  # Keep to 0.3, isotropic vibration on first axis
    structure.B22 = 0.3  # Keep to 0.3, isotropic vibration on second axis
    structure.B33 = 0.3  # Keep to 0.3, isotropic vibration on third axis
    structure.B12 = 0  # Keep at 0, anisotropic vibration
    structure.B13 = 0  # Keep at 0, anisotropic vibration
    structure.B23 = 0  # Keep at 0, anisotropic vibration

    pdf_params = dict(
        rmin=0,  # minimum value at which the pdf is evaluated
        rmax=30,  # maximum value at which the pdf is evaluated
        rstep=0.01,  # step-size of evaluation of pdf
        qmin=0.8,  # dont worry about this, keep at this value
        qmax=30,  # dont worry about this, keep at this value
        delta2=0,  # dont worry about this, keep at this value
        qdamp=0.01,  # dont worry about this, keep at this value
    )

    pdf_calculator = DebyePDFCalculator(**pdf_params)
    r, pdf = pdf_calculator(structure)
    r = torch.tensor(r, dtype=torch.float32)
    pdf = torch.tensor(pdf, dtype=torch.float32)
    pdf /= (torch.max(pdf) + 1e-12)

    return pdf


def calculate_pdf_from_adjecency_matrix(adj_matrix, atom_species):
    """
    Calculate pdf from adjecency matrix
    args:
        adj_matrix (np.array): adjecency matrix of size (n, n)
        atom_species (np.array): atom species of size (n,) containing their atomic weghts
    returns:
        pdf (np.array): array of evaluated points of the pdf function of shape (3000,)
    """
    point_cloud = multidimensional_scaling(adj_matrix)
    pdf = calculate_pdf(point_cloud, atom_species)
    return pdf


def calculate_pdf_batch(matrix_in, atom_species, pad_mask):
    """
    Generate pdf from batch of predictions
    args:
        matrix_in (torch.Tensor): batch of predictions of shape (batch_size, *)
                                  (can be both point cloud or adjecency matrix)
        atom_species (torch.Tensor): batch of atomic weights of shape (batch_size, n)
        pad_mask (torch.Tensor): batch of padding masks of shape (batch_size, n)
                                 (1 if atom is not padded, 0 if atom is padded)
    """
    pdfs = []
    if matrix_in.shape[2] != 3:  # If matrix_in is adjecency matrix
        for i in range(matrix_in.shape[0]):
            adj_matrix = matrix_in[i][:, pad_mask[i]][:, :, pad_mask[i]]
            species = atom_species[i][pad_mask[i]]
            pdfs.append(calculate_pdf_from_adjecency_matrix(adj_matrix[0], species))
    else:  # If matrix_in is point cloud
        for i in range(matrix_in.shape[0]):
            point_cloud = matrix_in[i][pad_mask[i]]
            species = atom_species[i][pad_mask[i]]
            pdfs.append(calculate_pdf(point_cloud, species))

    return torch.stack(pdfs)


def pearson_metric(predicted_pdf, target_pdf):
    """
    Calculate pearson correlation coefficient between predicted and target pdf
    args:
        predicted_pdf (torch.Tensor): predicted pdf of shape (batch_size, 3000)
        target_pdf (torch.Tensor): target pdf of shape (batch_size, 3000)
    returns:
        pearson (torch.Tensor): pearson correlation coefficient of shape (batch_size,)
    """
    pearson = []
    for i in range(predicted_pdf.shape[0]):
        pearson.append(
            pearsonr(predicted_pdf[i].detach().cpu().numpy(), target_pdf[i].detach().cpu().numpy())[0]
        )
    return torch.tensor(pearson, dtype=torch.float32)


def mse_metric(predicted_pdf, target_pdf):
    """
    Calculate mean squared error between predicted and target pdf
    args:
        predicted_pdf (torch.Tensor): predicted pdf of shape (batch_size, 3000)
        target_pdf (torch.Tensor): target pdf of shape (batch_size, 3000)
    returns:
        mse (torch.Tensor): mean squared error of shape (batch_size,)
    """
    return torch.mean((predicted_pdf - target_pdf) ** 2, dim=1)


def rwp_metric(predicted_pdf, true_pdf, sigmas=None):
    """
    Calculate the Reitveld Weighted Profile (RWP) metric
    args:
        predicted_pdf (torch.Tensor): predicted pdf of shape (batch_size, 3000)
        target_pdf (torch.Tensor): target pdf of shape (batch_size, 3000)
        sigmas (torch.Tensor): condfidence of predicted pdf, defaults to 1 (batch_size, 3000)
    returns:
        rwp (torch.Tensor): RWP metric of shape (batch_size,)
    """
    if sigmas is None:
        sigmas = torch.ones_like(predicted_pdf)
    sigmas_inv = 1 / sigmas
    diff_squarred = (predicted_pdf - true_pdf) ** 2
    true_squarred = true_pdf ** 2
    rwp = torch.sqrt((sigmas_inv * diff_squarred).sum(dim=1) / (sigmas_inv * true_squarred).sum(dim=1))

    return rwp
