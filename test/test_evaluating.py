import numpy as np
import pytest

from baby_pytorch.evaluating import evaluating
from baby_pytorch.nn import BatchNorm1d, Linear, MLP, Module, Tanh
from baby_pytorch.tensor import Tensor


class Model(Module):
    def __init__(self):
        super().__init__()
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return super().eval()

    def forward(self, x, training):
        return x * 2

    def parameters(self):
        return []

    def save_weights(self):
        return []

    def load_weights(self, weights):
        return None

    def __repr__(self):
        return "Model"


class Container(Module):
    def __init__(self, *parts):
        super().__init__()
        self.parts = list(parts)

    def forward(self, x, training):
        for part in self.parts:
            x = part(x)
        return x

    def parameters(self):
        return [
            parameter
            for part in self.parts
            for parameter in part.parameters()
        ]

    def save_weights(self):
        return [
            weight
            for part in self.parts
            for weight in part.save_weights()
        ]

    def load_weights(self, weights):
        return None

    def __repr__(self):
        return "Container"


def test_evaluating_temporarily_switches_model_to_evaluation_mode():
    model = Model()
    tensor = Tensor([1, 2, 3], requires_grad=True)

    with evaluating(model):
        assert model.training is False
        result = model(tensor)

    assert model.training is True
    assert model.eval_calls == 1
    assert result.requires_grad is False


def test_evaluating_preserves_existing_evaluation_mode():
    model = Model()
    model.eval()

    with evaluating(model):
        assert model.training is False

    assert model.training is False


def test_evaluating_restores_training_mode_after_exception():
    model = Model()

    with pytest.raises(RuntimeError, match="evaluation failed"):
        with evaluating(model):
            assert model.training is False
            raise RuntimeError("evaluation failed")

    assert model.training is True


def test_evaluating_can_be_nested():
    model = Model()

    with evaluating(model):
        assert model.training is False

        with evaluating(model):
            assert model.training is False

        assert model.training is False

    assert model.training is True


def test_evaluating_can_be_used_as_a_decorator():
    model = Model()

    @evaluating(model)
    def predict():
        assert model.training is False
        return "prediction"

    assert predict() == "prediction"
    assert model.training is True


def test_evaluating_controls_stateful_module_behavior():
    model = BatchNorm1d(2, momentum=1)
    tensor = Tensor([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
    original_mean = model.running_mean.clone()
    original_var = model.running_var.clone()

    with evaluating(model):
        result = model(tensor)

    assert model.training is True
    assert result.requires_grad is False
    np.testing.assert_array_equal(model.running_mean.data, original_mean.data)
    np.testing.assert_array_equal(model.running_var.data, original_var.data)


def test_evaluating_restores_child_module_modes():
    model = MLP(2, [3], 1, Tanh())

    with evaluating(model):
        assert all(child.training is False for child in model.children())

    assert model.training is True
    assert all(child.training is True for child in model.children())


def test_evaluating_restores_each_descendant_mode_individually():
    linear = Linear(2, 2)
    frozen = BatchNorm1d(2, momentum=1)
    inner = Container(linear, frozen)
    model = Container(inner)
    frozen.eval()
    tensor = Tensor([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
    original_mean = frozen.running_mean.clone()
    original_var = frozen.running_var.clone()

    with evaluating(model):
        assert model.training is False
        assert inner.training is False
        assert linear.training is False
        assert frozen.training is False
        model(tensor)

    assert model.training is True
    assert inner.training is True
    assert linear.training is True
    assert frozen.training is False

    model(tensor)

    np.testing.assert_array_equal(frozen.running_mean.data, original_mean.data)
    np.testing.assert_array_equal(frozen.running_var.data, original_var.data)
