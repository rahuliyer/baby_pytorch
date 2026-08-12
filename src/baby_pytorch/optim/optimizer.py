from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self):
        ...

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0.0
