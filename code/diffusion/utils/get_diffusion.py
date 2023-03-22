from diffusion_modules.GaussianDiffusion import GaussianDiffusion
from diffusion_modules.UniformCategoricalDiffusion import UniformCategoricalDiffusion


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
        n_categorical = args.n_categorical_vars
        n_vals = args.n_classes
        diffusion = UniformCategoricalDiffusion(*args)
    else:
        raise ValueError(f"Diffusion {args.diffusion} not implemented")
    return diffusion