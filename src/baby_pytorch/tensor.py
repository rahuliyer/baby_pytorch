import random
import string

import numpy as np

from baby_pytorch.functions import topo_sort
from baby_pytorch.functions import calculate_child_gradients
from baby_pytorch.no_grad import grad_enabled

class Tensor:
    def __init__(self,
                 data,
                 children=None,
                 op='',
                 requires_grad=False,
                 label='',
                 dtype=float):
        if children is not None and not grad_enabled():
            children = None
            op = ''
            requires_grad = False

        self.data = np.array(data, dtype=dtype)
        self.dtype=dtype
        self.children = [] if children is None else children
        self.op = op
        self.requires_grad = requires_grad
        self.grad = np.zeros(self.data.shape)
        self.ctx = {}

        if label != '':
            self.label = label
        else:
            suffix = ''.join(
                random.choice(string.ascii_lowercase + string.digits)
                for _ in range(4)
            )
            self.label = f'tensor_{suffix}'

    def __repr__(self):
        return f"<Tensor(data: {self.data} label: \"{self.label}\">"

    def __getTensor(self, value):
        if isinstance(value, Tensor):
            return value

        return Tensor(value, requires_grad=False)

    def __add__(self, other):
        other = self.__getTensor(other)

        out = Tensor(
                data=self.data + other.data,
                children=[self, other],
                op='+',
                requires_grad=self.requires_grad or other.requires_grad,
        )

        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self.__getTensor(other)

        out = Tensor(
                data=self.data - other.data,
                children=[self, other],
                op='-',
                requires_grad=self.requires_grad or other.requires_grad,
        )

        return out

    def __rsub__(self, other):
        other = self.__getTensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = self.__getTensor(other)

        out = Tensor(
                data=self.data * other.data,
                children=[self, other],
                op='*',
                requires_grad=self.requires_grad or other.requires_grad,
        )

        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        other = self.__getTensor(other)

        out = Tensor(
                data=self.data ** other.data,
                children=[self, other],
                op='**',
                requires_grad=self.requires_grad or other.requires_grad,
        )

        return out

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        other = self.__getTensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = self.__getTensor(other)
        return other.__truediv__(self)

    def log(self):
        out = Tensor(
                data=np.log(self.data),
                children=[self],
                op='log',
                requires_grad=self.requires_grad,
        )

        return out

    def log10(self):
        out = Tensor(
                data=np.log10(self.data),
                children=[self],
                op='log10',
                requires_grad=self.requires_grad,
        )

        return out

    def exp(self):
        out = Tensor(
                data=np.exp(self.data),
                children=[self],
                op='exp',
                requires_grad=self.requires_grad,
        )

        return out

    def __matmul__(self, other):
        out = Tensor(
            data=np.matmul(self.data, other.data),
            children=[self, other],
            op='matmul',
            requires_grad=self.requires_grad or other.requires_grad
        )

        return out

    def reshape(self, *shape):
        out = Tensor(
            data=self.data.reshape(*shape),
            children = [self],
            op='reshape',
            requires_grad=self.requires_grad
        )

        out.ctx["original_shape"] = self.data.shape

        return out

    def view(self, *shape):
        return self.reshape(*shape)

    def swapaxes(self, axis1, axis2):
        out = Tensor(
            data=self.data.swapaxes(axis1, axis2),
            children=[self],
            op='swapaxes',
            requires_grad=self.requires_grad
        )

        out.ctx["axis1"] = axis1
        out.ctx["axis2"] = axis2

        return out

    def T(self):
        out = Tensor(
            data=self.data.T,
            children=[self],
            op='T',
            requires_grad=self.requires_grad
        )

        return out

    def __getitem__(self, index):
        if isinstance(index, Tensor):
            index = index.data

        out = Tensor(
            data=self.data[index],
            children=[self],
            op='index',
            requires_grad=self.requires_grad
        )

        out.ctx["index"] = index

        return out

    def sum(self, dim=None, keepdims=False):
        out = Tensor(
            data=self.data.sum(axis=dim, keepdims=keepdims),
            children=[self],
            op='sum',
            requires_grad=self.requires_grad
        )

        out.ctx["keepdims"] = keepdims
        out.ctx["dim"] = dim

        return out

    def mean(self, dim=None, keepdims=False):
        out_data = self.data.sum(axis=dim, keepdims=keepdims)
        n = self.data.size // out_data.size
        out_data /= n

        out = Tensor(
            data=out_data,
            children=[self],
            op='mean',
            requires_grad=self.requires_grad
        )

        out.ctx["dim"] = dim
        out.ctx["keepdims"] = keepdims

        return out

    def var(self, dim=None, keepdims=False, correction=1):
        mean = self.mean(dim=dim, keepdims=True)
        reduced_element_count = self.data.size // mean.data.size
        squared_deviations = ((self - mean) ** 2).sum(
            dim=dim,
            keepdims=keepdims,
        )
        return squared_deviations / (reduced_element_count - correction)

    def std(self, dim=None, keepdims=False, correction=1):
        return self.var(
            dim=dim,
            keepdims=keepdims,
            correction=correction,
        ) ** 0.5

    def detach(self):
        out = Tensor(self.data, requires_grad=False, dtype=self.dtype)
        out.data = self.data

        return out

    def clone(self):
        # An untracked clone stays a leaf. This is especially important for
        # detach().clone() snapshots, which must not retain graph edges.
        if not self.requires_grad:
            return Tensor(self.data.copy(), dtype=self.dtype)

        return Tensor(self.data.copy(),
                      children=[self],
                      op='clone',
                      requires_grad=self.requires_grad,
                      dtype=self.dtype)

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
        return self

    def backward(self):
        if self.data.size != 1:
            raise ValueError("backward() only supported on scalar types")

        nodes = topo_sort(self)
        nodes.reverse()

        # We are going to follow the pytorch convention that gradients
        # at the leaf nodes accumulate across multiple backwards calls.
        # We need to clear the intermediate grads as these are read in the
        # chain rule implementation and leaving them in there would produce
        # incorrect results.
        for node in nodes:
            if node.children:
                node.grad = np.zeros(node.data.shape)

        # This is for a weird pytorch convention. If you call backward() on
        # a leaf node, the gradients accumulate (i.e 1 is added each time)
        # It's kinda weird because backward() on a lead doesn't really mean
        # anything.
        if self.children:
            self.grad = np.ones(self.data.shape)
        else:
            self.grad += np.ones(self.data.shape)

        for node in nodes:
            calculate_child_gradients(node)

    @property
    def shape(self):
        return self.data.shape
