import argparse
import torch
import torch_geometric as tg
import h5py
import glob
from tqdm import tqdm

def load_structure(path):

    with h5py.File(path, 'r') as file:
        edge_features = torch.tensor(file['edge_attributes'][:], dtype=torch.float32) # Edge attributes
        edge_indices = torch.tensor(file['edge_indices'][:], dtype=torch.long) # Edge (sparse) adjecency matrix
        node_features = torch.tensor(file['node_attributes'][:], dtype=torch.float32) # Node attributes

        r = file['r'][...] # Real space (x-axis)
        pdf = file['pdf'][...] # G(r) (y-axis)

    return edge_features, edge_indices, node_features, r, pdf

def pad_batches(array, max_length):
    """Adds padding over axis 0 to array"""
    padded_array = torch.zeros((max_length, *array.shape[1:]))
    padded_array[:array.shape[0]] = array
    return padded_array
    
def make_adjecency_matrix(edge_indices, edge_features, max_length):
    """Creates a dense adjecency matrix from a sparse adjecency matrix"""
    adjecency_matrix = torch.zeros((max_length, max_length))
    adjecency_matrix[edge_indices[0, :], edge_indices[1, :]] = edge_features
    return adjecency_matrix

def main(args):
    file_names = glob.glob(f'{args.data_dir}/*.h5')
    
    #Max number of nodes = 511
    #Number of node attributes = 7
    node_tensor = torch.zeros((len(file_names), 511, 7))
    adjecency_tensor = torch.zeros((len(file_names), 511, 511))
    pdf_tensor = torch.zeros((len(file_names), 3000))
    num_particles = torch.zeros(len(file_names))

    for i, p in tqdm(enumerate(file_names), total=len(file_names)):
        edge_features, edge_indices, node_features, r, pdf = load_structure(p)
        node_tensor[i] = pad_batches(node_features, 511)
        adjecency_tensor[i] = make_adjecency_matrix(edge_indices, edge_features, 511)
        pdf_tensor[i] = torch.from_numpy(pdf) 
        num_particles[i] = node_features.shape[0]

    print('Saving node features')
    torch.save(node_tensor, f'{args.save_dir}/node_tensor.pt')
    print('Saving number of particles')
    torch.save(num_particles, f'{args.save_dir}/num_particles.pt')
    print('Saving adjecency matrix')
    torch.save(adjecency_tensor, f'{args.save_dir}/adjecency_tensor.pt')
    print('Saving r')
    torch.save(r, f'{args.save_dir}/r_tensor.pt') # r is the same for all graphs
    print('Saving pdf')
    torch.save(pdf_tensor, f'{args.save_dir}/pdf_tensor.pt')
    print('Done')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='graphs_h5/')
    parser.add_argument('--save_dir', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)