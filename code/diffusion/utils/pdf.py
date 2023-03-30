"""
Functions for simulating pair distribution functions (PDFs) from predicted
structures.

@Author Thomas Christensen and Rasmus Pallisgaard
"""
import torch

from sklearn.manifold import MDS

from diffpy.structure import Structure, Atom
from diffpy.srreal.pdfcalculator import DebyePDFCalculator

from mendeleev import element
import multiprocessing as mp


# Task for parralelization in calculate_pdf
def task(inp):
    atom, xyz = inp
    atom = atom.item()
    return Atom(element(atom).symbol, xyz)


def multidimensional_scaling(adj_matrix):
    """
    Converet adjecency matrix to point cloud using multidimensional scaling
    args:
        adj_matrix (np.array): adjecency matrix of size (n, n)
    returns:
        point cloud (np.array): point cloud of size (n, 3)
    """
    # Symmetrise adjecency matrix
    _adj = torch.zeros_like(adj_matrix)
    triu_indices = torch.triu_indices(adj_matrix.shape[0], adj_matrix.shape[1], offset=1)
    _adj[triu_indices[0], triu_indices[1]] = adj_matrix[triu_indices[0], triu_indices[1]]
    _adj[triu_indices[1], triu_indices[0]] = adj_matrix[triu_indices[0], triu_indices[1]]

    mds = MDS(
        n_components=3,
        metric=True,
        max_iter=3000,
        eps=1e-9,
        dissimilarity="precomputed",
        n_jobs=4
    )

    point_cloud = mds.fit_transform(_adj)
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
    atom_species = atom_species.int()

    with mp.Pool() as pool:
        for atom in pool.imap(task, zip(atom_species, point_cloud)):
            structure.append(atom)

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
    # Invert padding mask, since we want to keep the atoms that are not padded
    pad_mask = torch.logical_not(pad_mask.bool())
    pdfs = []
    if matrix_in.shape[2] != 3:  # If matrix_in is adjecency matrix
        for i in range(matrix_in.shape[0]):
            adj_matrix = matrix_in[i][pad_mask[i]][:, pad_mask[i]]
            species = atom_species[i][pad_mask[i]]
            pdfs.append(calculate_pdf_from_adjecency_matrix(adj_matrix, species))
    else:  # If matrix_in is point cloud
        for i in range(matrix_in.shape[0]):
            point_cloud = matrix_in[i][pad_mask[i]]
            species = atom_species[i][pad_mask[i]]
            pdfs.append(calculate_pdf(point_cloud, species))

    return torch.stack(pdfs)
