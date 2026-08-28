from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self):
        ...

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros(param.shape)
