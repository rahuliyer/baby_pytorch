import numpy as np

from baby_pytorch.tensor import Tensor

def tanh(x):
    out = Tensor(
        data=np.tanh(x.data),
        op='tanh',
        children=[x],
        requires_grad=x.requires_grad
    )

    return out

def sigmoid(x):
    out = Tensor(
        data=np.exp(-np.logaddexp(0, -x.data)),
        op='sigmoid',
        children=[x],
        requires_grad=x.requires_grad
    )

    return out

def relu(x):
    out = Tensor(
        data=np.maximum(0, x.data),
        op='relu',
        children=[x],
        requires_grad=x.requires_grad
    )

    return out
