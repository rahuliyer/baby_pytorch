import random

from baby_pytorch.tensor import Tensor

class Neuron:
    def __init__(self, in_size):
        self.weights = [
            Tensor(random.uniform(-1, 1), requires_grad=True)
                  for _ in range(in_size)
        ]
        self.bias = Tensor(random.uniform(-1, 1), requires_grad=True)

    def __call__(self, input):
        if len(input) != len(self.weights):
            raise ValueError("Input size must match the number of weights.")

        result = sum(w * i for w, i in zip(self.weights, input)) + self.bias

        return result

    def parameters(self):
        return self.weights + [self.bias]
