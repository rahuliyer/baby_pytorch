from abc import ABC, abstractmethod


class Module(ABC):
    @abstractmethod
    def forward(self, x, training):
        ...

    def __call__(self, x, training=True):
        out = self.forward(x, training)
        return out if training else out.detach()

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
