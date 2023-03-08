"""
Functions to get the model given args

@Author Thomas Christensen and Rasmus Pallisgaard
"""
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
    if args.diffusion == "ddpm":
        diffusion = DDPM()
    else:
        raise ValueError(f"Diffusion {args.diffusion} not implemented")
    return diffusion