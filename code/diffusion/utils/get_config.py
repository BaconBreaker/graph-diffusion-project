"""
Functions to get the model given args

@Author Thomas Christensen and Rasmus Pallisgaard
"""

from diffusion_modules.GaussianDiffusion import GaussianDiffusion
from diffusion_modules.UniformCategoricalDiffusion import UniformCategoricalDiffusion
from networks.SelfAttentionNetwork import (SelfAttentionNetwork,
    self_attention_posttransform, self_attention_pretransform)
from networks.ConditionalUNet import ConditionalUNet


def get_model(args):
    """
    Returns the model given the args
    args:
        args: argparse object
    returns:
        model: torch.nn.Module
    """
    if args.model == "conditional_unet":
        model = ConditionalUNet(num_classes=args.num_classes).to(args.device)
        pretransform = None
        posttransform = None
    elif args.model == "self_attention":
        model = SelfAttentionNetwork(num_classes=args.num_classes, input_size=args.image_size).to(args.device)
        pretransform = self_attention_pretransform
        posttransform = self_attention_posttransform
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
        noise_shape = args.noise_shape
        model_target = args.model_target
        diffusion = GaussianDiffusion(noise_shape, model_target)
    elif args.diffusion == "uniform_categorical":
        n_categorical_vars = args.n_categorical_vars
        n_values = args.n_values
        diffusion = UniformCategoricalDiffusion(n_categorical_vars, n_values)
    else:
        raise ValueError(f"Diffusion {args.diffusion} not implemented")
    return diffusion
