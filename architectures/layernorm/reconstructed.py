import torch

def build(spec, parameters):
    normalized_shape = tuple(spec['config']['normalized_shape'])
    eps = float(spec['config']['eps'])
    gamma = parameters['gamma']
    beta = parameters['beta']

    class LayerNormComponent(torch.nn.Module):
        def __init__(self, normalized_shape, eps, gamma, beta):
            super().__init__()
            self.normalized_shape = normalized_shape
            self.eps = eps
            self.gamma = gamma
            self.beta = beta

        def forward(self, x):
            axes = tuple(range(x.dim() - len(self.normalized_shape), x.dim()))
            mean = x.mean(dim=axes, keepdim=True)
            center = x - mean
            variance = (center * center).mean(dim=axes, keepdim=True)
            normalized = center / torch.sqrt(variance + self.eps)
            y = normalized * self.gamma + self.beta
            return y

    return LayerNormComponent(normalized_shape, eps, gamma, beta)
