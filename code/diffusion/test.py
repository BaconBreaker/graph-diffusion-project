# import argparse
import logging
# import pytorch_lightning as pl
import torch
from torch import nn
# from train import train
from main import parse_args, check_args
from utils.get_config import get_model, get_diffusion
from DiffusionWrapper import DiffusionWrapper
from dataloader import MoleculeDataModule
from callbacks import get_callbacks
from utils.plots import save_graph
from utils.metrics import get_metrics
from networks.graph_transformer_model import GraphTransformer


logging.basicConfig(format='%(asctime)s - %(levelname)s %(message)s',
                    level=logging.INFO,
                    datefmt="%I:%M:%S")
logger = logging.getLogger(__name__)

# python test.py --dataset_path ../../graphs_fixed_num_135/ --run_name 123 --max_epochs 1000 --check_val_every_n_epoch 10 --batch_size 4 --tensors_to_diffuse edge_sequence --pad_length 135 --diffusion_timesteps 3 --num_workers 8 --log_every_n_steps 10 --disable_carbon_tracker


# def test(args):
#     n_layers = 5
#     input_dims = {"X": 2, "E": 30, "y": 4}
#     hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 128}
#     hidden_dims = {'dx': 256,
#                    'de': 64,
#                    'dy': 64,
#                    'n_head': 8,
#                    'dim_ffX': 256,
#                    'dim_ffE': 128,
#                    'dim_ffy': 128}
#     output_dims = {"X": 2, "E": 30, "y": 4}
#     act_fn_in = nn.ReLU()
#     act_fn_out = nn.ReLU()
#
#     model, pretransform, posttransform = get_model(args)
#     transformer = GraphTransformer(n_layers,
#                                    input_dims,
#                                    hidden_mlp_dims,
#                                    hidden_dims,
#                                    output_dims,
#                                    act_fn_in,
#                                    act_fn_out)
#
#     n_nodes = 135
#     node_mask = torch.ones(4, n_nodes)
#     out = transformer(torch.randn(4, 135, 3),
#                       torch.randn(4, 135, 135, 10),
#                       torch.randn(4, 4),
#                       node_mask)
#
#     print(f"out type: {type(out)}")
#
#     dataloader = MoleculeDataModule(args, transform=None)
#     dataloader.prepare_data()
#     dataloader.setup()
#
#     # molecule_info = dataloader.molecule_info
#
#     ex_batch = next(iter(dataloader.train_dataloader()))
#     print(ex_batch.keys())
#     print(ex_batch["node_features"].shape)
#     print(ex_batch["adj_matrix"].shape)
#     print(ex_batch["node_features"][0, :, 0])
#
#     transform_batch = pretransform(ex_batch)
#     print(transform_batch.keys())
#     print(transform_batch["node_features"].shape)
#     print(transform_batch["edge_features"].shape)
#
#     print("OK.")


def test2(args):
    model, pretransform, posttransform = get_model(args)
    diffusion_model = get_diffusion(args)
    dataloader = MoleculeDataModule(args=args, transform=pretransform)
    metrics = get_metrics(args)
    diffusion = DiffusionWrapper(denoising_fn=model,
                                 diffusion_model=diffusion_model, lr=args.lr,
                                 posttransform=posttransform, metrics=metrics)
    callbacks = get_callbacks(args)

    dataloader.prepare_data()
    dataloader.setup()

    train_dl = dataloader.train_dataloader()
    ex_batch = next(iter(train_dl))
    save_graph(ex_batch,
               t=1,
               run_name=args.run_name + "_test_generation",
               post_process=posttransform)

    _ = diffusion_model.sample(model, ex_batch, save_output=True,
                                         post_process=posttransform)

    print("OK.")


def main():
    args = parse_args()
    check_args(args)
    test2(args)


if __name__ == "__main__":
    main()
