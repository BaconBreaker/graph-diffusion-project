"""
Functions to get the model given args

@Author Thomas Christensen and Rasmus Pallisgaard
"""

from diffusion_modules.GaussianDiffusion import GaussianDiffusion
from diffusion_modules.UniformCategoricalDiffusion import UniformCategoricalDiffusion
from networks.SelfAttentionNetwork import (
    SelfAttentionNetwork,
    self_attention_posttransform,
    self_attention_pretransform
)
from networks.ConditionalUNet import ConditionalUNet
from networks.EquivariantNetwork import (
    EquivariantNetwork,
    equivariant_posttransform,
    equivariant_pretransform
)
# from networks.DiGress import digress_pretransform
from networks.EdgeAugmentedTransformer import (
    EdgeAugmentedGraphTransformer,
    EAGTPretransform,
    EAGTPosttransform
)


def get_model(args):
    """
    Returns the model given the args
    args:
        args: argparse object
    returns:
        model: torch.nn.Module
    """
    if args.model == "conditional_unet":
        model = ConditionalUNet(num_classes=args.num_classes)
        pretransform = None
        posttransform = None
    elif args.model == "self_attention":
        model = SelfAttentionNetwork(args)
        pretransform = self_attention_pretransform
        posttransform = self_attention_posttransform
    elif args.model == "equivariant":
        model = EquivariantNetwork(args)
        pretransform = equivariant_pretransform
        posttransform = equivariant_posttransform
    elif args.model == "digress":
        model = None
        pretransform = digress_pretransform
        posttransform = None
    elif args.model == "eagt":
        model = EdgeAugmentedGraphTransformer(args)
        pretransform = EAGTPretransform
        posttransform = EAGTPosttransform
    # elif args.model == "digress":
    #     model = None
    #     pretransform = digress_pretransform
    #     posttransform = None
    else:
        raise ValueError(f"Model {args.model} not implemented")
    return model, pretransform, posttransform


def get_diffusion(args):
    """
    Returns the diffusion model given the args
    args:
        args: argparse object
    returns:
        diffusion: torch.nn.Module
    """
    if args.diffusion == "gaussian":
        diffusion = GaussianDiffusion(args)
    elif args.diffusion == "uniform_categorical":
        n_categorical_vars = args.n_categorical_vars
        n_values = args.n_values
        diffusion = UniformCategoricalDiffusion(n_categorical_vars, n_values)
    else:
        raise ValueError(f"Diffusion {args.diffusion} not implemented")
    return diffusion
