from baby_pytorch.optim import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters, lr=lr)

    def step(self):
        for param in self.parameters:
            if param.requires_grad:
                param.data -= self.lr * param.grad