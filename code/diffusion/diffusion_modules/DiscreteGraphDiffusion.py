from diffusion_modules.BaseDiffusion import Diffusion
from diffusion_modules.UniformCategoricalDiffusion import UniformCategoricalDiffusion


class DiscreteGraphDiffusion(Diffusion):
    def __init__(self, n_node_classes, n_edge_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_node_classes = n_node_classes
        self.n_edge_classes = n_edge_classes
        self.node_diffusion = UniformCategoricalDiffusion(n_categorical_vars=1, n_classes=self.n_node_classes)
        self.edge_diffusion = UniformCategoricalDiffusion(n_categorical_vars=self.n_edge_classes, n_classes=self.n_edge_classes)

    def diffuse(self, inp, t):
        x0, e0 = inp
        x = self.node_diffusion.diffuse(x0, t)
        e = self.edge_diffusion.diffuse(e0, t)
        return x, e

    def sample(self, inp, t):
        x0, e0 = inp
        x = self.node_diffusion.sample(x0, t)
        e = self.edge_diffusion.sample(e0, t)
        return x, e