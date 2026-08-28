import numpy as np

from baby_pytorch.nn.module import Module
from baby_pytorch.tensor import Tensor


class BatchNorm1d(Module):
    def __init__(self, fan_in, eps=1e-5, momentum=0.01):
        self.fan_in = fan_in
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones(fan_in), requires_grad=True)
        self.beta = Tensor(np.zeros(fan_in), requires_grad=True)
        self.running_mean = Tensor(np.zeros(fan_in))
        self.running_var = Tensor(np.ones(fan_in))

    def forward(self, x, training):
        if len(x.shape) not in (2, 3):
            raise ValueError("BatchNorm1d expects a 2-D or 3-D input.")
        if x.shape[1] != self.fan_in:
            raise ValueError(
                f"Expected {self.fan_in} channels, received {x.shape[1]}."
            )

        reduction_dims = 0 if len(x.shape) == 2 else (0, 2)
        broadcast_shape = (
            (1, self.fan_in)
            if len(x.shape) == 2
            else (1, self.fan_in, 1)
        )

        if training:
            mean = x.mean(dim=reduction_dims, keepdims=True)
            var = x.var(dim=reduction_dims, keepdims=True)
            normalized = (x - mean) / (var + self.eps) ** 0.5

            self.running_mean = (
                self.running_mean * (1 - self.momentum)
                + mean.reshape(self.fan_in).detach() * self.momentum
            ).detach()
            self.running_var = (
                self.running_var * (1 - self.momentum)
                + var.reshape(self.fan_in).detach() * self.momentum
            ).detach()
        else:
            normalized = (
                (x - self.running_mean.reshape(broadcast_shape))
                / (self.running_var.reshape(broadcast_shape) + self.eps) ** 0.5
            )

        return (
            normalized * self.gamma.reshape(broadcast_shape)
            + self.beta.reshape(broadcast_shape)
        )

    def parameters(self):
        return [self.gamma, self.beta]

    def save_weights(self):
        parameters = [
            parameter.detach().clone()
            for parameter in self.parameters()
        ]
        running_statistics = [
            self.running_mean.clone(),
            self.running_var.clone(),
        ]
        return parameters + running_statistics

    def load_weights(self, weights):
        state = [
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
        ]
        if len(weights) != len(state):
            raise ValueError(
                f"Expected {len(state)} weights, received {len(weights)}."
            )

        for tensor, weight in zip(state, weights):
            if tensor.shape != weight.shape:
                raise ValueError(
                    f"Expected weight shape {tensor.shape}, "
                    f"received {weight.shape}."
                )
            tensor.data[...] = weight.data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}: self.gamma.shape={self.gamma.shape}, "
            f"self.beta.shape={self.beta.shape}"
        )
