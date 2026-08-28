import numpy as np
import pytest

from baby_pytorch import Tensor, grad_enabled, no_grad
from baby_pytorch.activation_functions import tanh


def test_no_grad_disables_and_restores_gradient_tracking():
    tensor = Tensor([1.0, 2.0], requires_grad=True)

    assert grad_enabled()
    with no_grad():
        assert not grad_enabled()
        result = tensor * 2

    assert grad_enabled()
    np.testing.assert_array_equal(result.data, [2.0, 4.0])
    assert not result.requires_grad
    assert result.children == []
    assert result.op == ""
    assert tensor.requires_grad


def test_no_grad_restores_tracking_after_an_exception():
    with pytest.raises(RuntimeError, match="boom"):
        with no_grad():
            assert not grad_enabled()
            raise RuntimeError("boom")

    assert grad_enabled()
    tensor = Tensor(2.0, requires_grad=True)
    assert (tensor * 3).requires_grad


def test_no_grad_can_be_nested_and_reused():
    context = no_grad()

    with context:
        assert not grad_enabled()
        with context:
            assert not grad_enabled()
        assert not grad_enabled()

    assert grad_enabled()
    with context:
        assert not grad_enabled()
    assert grad_enabled()


def test_no_grad_can_decorate_repeated_function_calls():
    tensor = Tensor(2.0, requires_grad=True)

    @no_grad()
    def double(value):
        assert not grad_enabled()
        return value * 2

    first = double(tensor)
    second = double(tensor)

    assert not first.requires_grad
    assert not second.requires_grad
    assert first.children == []
    assert second.children == []
    assert grad_enabled()


@pytest.mark.parametrize(
    "operation",
    [
        lambda tensor: tensor + 2,
        lambda tensor: tensor - 2,
        lambda tensor: tensor * 2,
        lambda tensor: tensor ** 2,
        lambda tensor: tensor / 2,
        lambda tensor: tensor.log(),
        lambda tensor: tensor.log10(),
        lambda tensor: tensor.exp(),
        lambda tensor: tensor @ Tensor([[1.0], [2.0]]),
        lambda tensor: tensor.reshape(1, 2),
        lambda tensor: tensor.swapaxes(0, 0),
        lambda tensor: tensor.T(),
        lambda tensor: tensor[0],
        lambda tensor: tensor.sum(),
        lambda tensor: tensor.mean(),
        lambda tensor: tensor.var(),
        lambda tensor: tensor.std(),
        lambda tensor: tensor.clone(),
        lambda tensor: tanh(tensor),
    ],
)
def test_no_grad_makes_operation_results_untracked_leaves(operation):
    tensor = Tensor([1.0, 2.0], requires_grad=True)

    with no_grad():
        result = operation(tensor)

    assert not result.requires_grad
    assert result.children == []
    assert result.op == ""


def test_no_grad_preserves_explicitly_trainable_leaf_creation():
    with no_grad():
        parameter = Tensor([1.0, 2.0], requires_grad=True)

    assert parameter.requires_grad
    assert parameter.children == []


def test_no_grad_does_not_disable_backward_on_an_existing_graph():
    tensor = Tensor(2.0, requires_grad=True)
    result = tensor * 3

    with no_grad():
        result.backward()

    assert tensor.grad == pytest.approx(3.0)


def test_enabling_grad_on_a_no_grad_result_does_not_reconnect_its_source():
    source = Tensor([1.0, 2.0], requires_grad=True)

    with no_grad():
        result = source * 2

    result.requires_grad_()
    (result * 3).sum().backward()

    np.testing.assert_array_equal(result.grad, [3.0, 3.0])
    np.testing.assert_array_equal(source.grad, [0.0, 0.0])
    assert result.children == []
