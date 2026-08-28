import numpy as np
import pytest

from baby_pytorch.nn import Module
from baby_pytorch.tensor import Tensor


class Double(Module):
    def __init__(self):
        super().__init__()
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


def test_module_modules_yields_self_for_a_leaf():
    module = Double()

    assert list(module.modules()) == [module]


def test_module_can_switch_between_training_and_evaluation_modes():
    module = Double()

    assert module.training is True
    assert module.eval() is module
    assert module.training is False
    assert module.train() is module
    assert module.training is True


def test_module_default_call_uses_its_current_mode():
    module = Double()
    tensor = Tensor([1, 2, 3], requires_grad=True)
    module.eval()

    result = module(tensor)

    assert module.last_training_value is False
    assert not result.requires_grad


def test_module_call_does_not_accept_a_training_override():
    module = Double()
    tensor = Tensor([1, 2, 3], requires_grad=True)

    with pytest.raises(TypeError, match="unexpected keyword argument 'training'"):
        module(tensor, training=False)


def test_train_rejects_non_boolean_modes():
    module = Double()

    with pytest.raises(ValueError, match="must be a boolean"):
        module.train(1)


def test_module_call_detaches_results_during_inference():
    module = Double()
    tensor = Tensor([1, 2, 3], requires_grad=True)
    module.eval()

    result = module(tensor)

    np.testing.assert_array_equal(result.data, [2, 4, 6])
    assert module.last_training_value is False
    assert not result.requires_grad
    assert result.children == []

    (result * 3).sum().backward()
    np.testing.assert_array_equal(tensor.grad, [0, 0, 0])
