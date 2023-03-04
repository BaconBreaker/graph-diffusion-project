"""
Functions to get the model given args

@Author Thomas Christensen and Rasmus Pallisgaard
"""
from networks.SelfAttentionNetwork import SelfAttentionNetwork
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
    elif args.model == "self_attention":
        model = SelfAttentionNetwork(num_classes=args.num_classes, input_size=args.image_size).to(args.device)
    else:
        raise ValueError(f"Model {args.model} not implemented")
    return model
