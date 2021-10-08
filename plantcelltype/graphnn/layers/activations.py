from torch import sigmoid
from torch.nn.functional import relu, leaky_relu, elu, silu, log_softmax


activations = {'relu': relu,
               'leaky_relu': leaky_relu,
               'elu': elu,
               'sigmoid': sigmoid,
               'silu': silu,
               'log_softmax': log_softmax,
               'none': lambda x: x}
