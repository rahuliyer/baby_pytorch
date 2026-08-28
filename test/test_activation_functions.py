from baby_pytorch.tensor import Tensor
from baby_pytorch.activation_functions import tanh, sigmoid, relu

import pytest

def test_tanh():
    x = Tensor(2)

    y = tanh(x)

    assert y.data == pytest.approx(0.9640275800758169)
    assert len(y.children) == 1
    assert x in y.children
    assert y.op == 'tanh'

def test_sigmoid():
    x = Tensor(2)

    y = sigmoid(x)

    assert y.data == pytest.approx(0.8807970779778823)
    assert len(y.children) == 1
    assert x in y.children
    assert y.op == 'sigmoid'

def test_sigmoid_negative_and_zero():
    assert sigmoid(Tensor(0.0)).data == pytest.approx(0.5)
    assert sigmoid(Tensor(-2.0)).data == pytest.approx(0.11920292202211755)

def test_relu():
    x = Tensor(2)

    y = relu(x)

    assert y.data == 2
    assert len(y.children) == 1
    assert x in y.children
    assert y.op == 'relu'

def test_relu_at_zero_and_negative():
    x = Tensor(-2.0, requires_grad=True)
    y = relu(x)

    y.backward()

    assert x.grad == 0
