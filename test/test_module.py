import numpy as np
import pytest

from baby_pytorch.nn import Module
from baby_pytorch.tensor import Tensor


class Double(Module):
    def __init__(self):
        self.last_training_value = None

    def forward(self, x, training):
        self.last_training_value = training
        return x * 2

    def parameters(self):
        return []

    def save_weights(self):
        return []

    def load_weights(self, weights):
        return None

    def __repr__(self):
        return "Double"


def test_module_requires_its_abstract_interface():
    with pytest.raises(TypeError):
        Module()


def test_module_call_uses_training_mode_by_default():
    module = Double()
    tensor = Tensor([1, 2, 3], requires_grad=True)

    result = module(tensor)

    np.testing.assert_array_equal(result.data, [2, 4, 6])
    assert module.last_training_value is True
    assert result.requires_grad


def test_module_call_detaches_results_during_inference():
    module = Double()
    tensor = Tensor([1, 2, 3], requires_grad=True)

    result = module(tensor, training=False)

    np.testing.assert_array_equal(result.data, [2, 4, 6])
    assert module.last_training_value is False
    assert not result.requires_grad
    assert result.children == []

    (result * 3).sum().backward()
    np.testing.assert_array_equal(tensor.grad, [0, 0, 0])
