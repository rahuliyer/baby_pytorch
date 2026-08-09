import random
import string

from baby_pytorch.functions import topo_sort
from baby_pytorch.functions import calculate_child_gradients

class Value:
    def __init__(self, data, children=None, op='', requires_grad=True, label=''):
        self.data = data
        self.children = [] if children is None else children
        self.op = op
        self.requires_grad = requires_grad
        self.grad = 0.0

        if label != '':
            self.label = label
        else:
            suffix = ''.join(
                random.choice(string.ascii_lowercase + string.digits)
                for _ in range(4)
            )
            self.label = f'val_{suffix}'

    def __repr__(self):
        return f"<Value(data: {self.data} label: \"{self.label}\">"

    def __getValue(self, val):
        return Value(val, requires_grad=False) if not isinstance(val, Value) else val

    def __add__(self, other):
        other = self.__getValue(other)

        out = Value(
                data=self.data + other.data,
                children=[self, other],
                op='+',
                requires_grad=self.requires_grad or other.requires_grad,
        )

        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self.__getValue(other)

        out = Value(
                data=self.data - other.data,
                children=[self, other],
                op='-',
                requires_grad=self.requires_grad or other.requires_grad,
        )

        return out

    def __rsub__(self, other):
        other = self.__getValue(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = self.__getValue(other)

        out = Value(
                data=self.data * other.data,
                children=[self, other],
                op='*',
                requires_grad=self.requires_grad or other.requires_grad,
        )

        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        other = self.__getValue(other)

        out = Value(
                data=self.data ** other.data,
                children=[self, other],
                op='**',
                requires_grad=self.requires_grad or other.requires_grad,
        )

        return out

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        other = self.__getValue(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = self.__getValue(other)
        return other.__truediv__(self)

    def backward(self):
        nodes = topo_sort(self)
        nodes.reverse()

        # We are going to follow the pytorch convention that gradients
        # at the leaf nodes accumulate across multiple backwards calls. 
        # We need to clear the intermediate grads as these are read in the
        # chain rule implementation and leaving them in there would produce
        # incorrect results.
        for node in nodes:
            if node.children:
                node.grad = 0.0

        # This is for a weird pytorch convention. If you call backward() on
        # a leaf node, the gradients accumulate (i.e 1 is added each time)
        # It's kinda weird because backward() on a lead doesn't really mean
        # anything.
        if self.children:
            self.grad = 1.0
        else:
            self.grad += 1.0

        for node in nodes:
            calculate_child_gradients(node)
