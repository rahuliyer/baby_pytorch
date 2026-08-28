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
        if training:
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            normalized = (x - mean) / (var + self.eps) ** 0.5

            self.running_mean = (
                self.running_mean * (1 - self.momentum)
                + mean.detach() * self.momentum
            ).detach()
            self.running_var = (
                self.running_var * (1 - self.momentum)
                + var.detach() * self.momentum
            ).detach()
        else:
            normalized = (
                (x - self.running_mean)
                / (self.running_var + self.eps) ** 0.5
            )

        return normalized * self.gamma + self.beta

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
        self.gamma = weights[0].detach().clone().requires_grad_()
        self.beta = weights[1].detach().clone().requires_grad_()
        self.running_mean = weights[2].detach().clone()
        self.running_var = weights[3].detach().clone()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}: self.gamma.shape={self.gamma.shape}, "
            f"self.beta.shape={self.beta.shape}"
        )
