from baby_pytorch.activation_functions import tanh
from baby_pytorch.nn.module import Module


class Tanh(Module):
    def forward(self, x, training):
        self.out = tanh(x)
        return self.out

    def parameters(self):
        return []

    def save_weights(self):
        return []

    def load_weights(self, weights):
        return None

    def __repr__(self):
        return self.__class__.__name__
