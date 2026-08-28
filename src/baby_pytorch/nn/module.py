from abc import ABC, abstractmethod


class Module(ABC):
    def __init__(self):
        self.training = True

    @abstractmethod
    def forward(self, x, training):
        ...

    def __call__(self, x):
        out = self.forward(x, self.training)
        return out if self.training else out.detach()

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("Training mode must be a boolean.")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def children(self):
        for value in vars(self).values():
            yield from self._modules_in(value)

    def _modules_in(self, value):
        if isinstance(value, Module):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from self._modules_in(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from self._modules_in(item)

    @abstractmethod
    def parameters(self):
        ...

    @abstractmethod
    def save_weights(self):
        ...

    @abstractmethod
    def load_weights(self, weights):
        ...

    @abstractmethod
    def __repr__(self):
        ...
