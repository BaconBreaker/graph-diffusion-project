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
        model = SelfAttentionNetwork(args).to(args.device)
        pretransform = self_attention_pretransform
        posttransform = self_attention_posttransform
    else:
        raise ValueError(f"Model {args.model} not implemented")
    return model, pretransform, posttransform
