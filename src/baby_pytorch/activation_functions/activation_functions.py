import math
from baby_pytorch.tensor import Tensor

def tanh(x):
    out = Tensor(
        data=math.tanh(x.data),
        op='tanh',
        children=[x],
        requires_grad=x.requires_grad
    )

    return out

def sigmoid(x):
    if x.data > 0:
        val = 1.0 / (1 + math.exp(-1 * x.data))
    else:
        e_x = math.exp(x.data)
        val = e_x / (1 + e_x)

    out = Tensor(
        data=val,
        op='sigmoid',
        children=[x],
        requires_grad=x.requires_grad
    )

    return out

def relu(x):
    out = Tensor(
        data=max(0, x.data),
        op='relu',
        children=[x],
        requires_grad=x.requires_grad
    )

    return out