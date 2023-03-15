from diffusion_modules.BaseDiffusion import Diffusion
from diffusion_modules.UniformCategoricalDiffusion import UniformCategoricalDiffusion


class DiscreteGraphDiffusion(Diffusion):
    def __init__(self, n_node_classes, n_edge_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_node_classes = n_node_classes
        self.n_edge_classes = n_edge_classes
        self.node_diffusion = UniformCategoricalDiffusion(n_categorical_vars=1, n_classes=self.n_node_classes)
        self.edge_diffusion = UniformCategoricalDiffusion(n_categorical_vars=1, n_classes=self.n_edge_classes)

    
