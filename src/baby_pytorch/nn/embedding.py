import numpy as np

from baby_pytorch.nn.module import Module
from baby_pytorch.tensor import Tensor


class Embedding(Module):
    def __init__(self, in_len, num_dims):
        self.in_len = in_len
        self.num_dims = num_dims
        self.embedding = Tensor(
            np.random.randn(in_len, num_dims),
            requires_grad=True,
        )

    def forward(self, index, training):
        return self.embedding[index]

    def parameters(self):
        return [self.embedding]

    def save_weights(self):
        return [parameter.detach().clone() for parameter in self.parameters()]

    def load_weights(self, weights):
        if len(weights) != 1:
            raise ValueError(f"Expected 1 weight, received {len(weights)}.")
        if self.embedding.shape != weights[0].shape:
            raise ValueError(
                f"Expected weight shape {self.embedding.shape}, "
                f"received {weights[0].shape}."
            )
        self.embedding.data[...] = weights[0].data

    def __repr__(self):
        return f"{self.__class__.__name__}: shape({self.embedding.shape})"
