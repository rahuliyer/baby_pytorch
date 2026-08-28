import numpy as np

from baby_pytorch.nn.module import Module
from baby_pytorch.tensor import Tensor


class Linear(Module):
    def __init__(self, fan_in, fan_out, bias=True):
        self.weights = Tensor(
            np.random.randn(fan_in, fan_out),
            requires_grad=True,
        )
        self.bias = (
            Tensor(np.zeros(fan_out), requires_grad=True)
            if bias
            else None
        )

    def forward(self, x, training):
        logits = x @ self.weights
        if self.bias is not None:
            logits = logits + self.bias
        return logits

    def parameters(self):
        parameters = [self.weights]
        if self.bias is not None:
            parameters.append(self.bias)
        return parameters

    def save_weights(self):
        return [parameter.detach().clone() for parameter in self.parameters()]

    def load_weights(self, weights):
        parameters = self.parameters()
        if len(weights) != len(parameters):
            raise ValueError(
                f"Expected {len(parameters)} weights, received {len(weights)}."
            )

        for parameter, weight in zip(parameters, weights):
            if parameter.shape != weight.shape:
                raise ValueError(
                    f"Expected weight shape {parameter.shape}, "
                    f"received {weight.shape}."
                )
            parameter.data[...] = weight.data

    def __repr__(self):
        bias_shape = None if self.bias is None else self.bias.shape
        return (
            f"{self.__class__.__name__}: weights: {self.weights.shape} "
            f"bias: {bias_shape}"
        )
