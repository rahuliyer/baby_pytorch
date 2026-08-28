import math

import numpy as np
import pytest

from baby_pytorch.nn import calculate_gain, kaiming_normal_
from baby_pytorch.tensor import Tensor


@pytest.mark.parametrize(
    ("nonlinearity", "expected"),
    [
        ("relu", math.sqrt(2.0)),
        ("sigmoid", 1.0),
        ("tanh", 5.0 / 3.0),
    ],
)
def test_calculate_gain_supports_expected_nonlinearities(
    nonlinearity,
    expected,
):
    assert calculate_gain(nonlinearity) == pytest.approx(expected)


@pytest.mark.parametrize("nonlinearity", ["leaky_relu", "linear", "ReLU", None])
def test_calculate_gain_rejects_unsupported_nonlinearities(nonlinearity):
    with pytest.raises(ValueError, match="Unsupported nonlinearity"):
        calculate_gain(nonlinearity)


@pytest.mark.parametrize("fan_in", [0, -1, 1.5, "2", True, np.bool_(False)])
def test_kaiming_normal_rejects_invalid_fan_in(fan_in):
    with pytest.raises(ValueError, match="fan_in must be a positive integer"):
        kaiming_normal_(Tensor(np.zeros(4)), fan_in, "relu")


def test_kaiming_normal_uses_gain_and_fan_in_for_scaling():
    np.random.seed(2147483647)
    tensor = Tensor(np.zeros(200_000))

    kaiming_normal_(tensor, fan_in=25, nonlinearity="tanh")

    expected_std = calculate_gain("tanh") / math.sqrt(25)
    assert tensor.data.mean() == pytest.approx(0.0, abs=0.003)
    assert tensor.data.std() == pytest.approx(expected_std, rel=0.01)


def test_kaiming_normal_updates_the_original_tensor_and_backing_array():
    tensor = Tensor(np.zeros((3, 4)), requires_grad=True)
    original_data = tensor.data

    result = kaiming_normal_(tensor, fan_in=np.int64(3), nonlinearity="relu")

    assert result is tensor
    assert tensor.data is original_data
    assert tensor.shape == (3, 4)
    assert np.any(tensor.data != 0.0)


def test_kaiming_normal_preserves_gradients_and_graph_metadata():
    source = Tensor([1.0, 2.0], requires_grad=True, label="source")
    tensor = source * 2.0
    tensor.label = "derived"
    tensor.ctx["sentinel"] = object()
    tensor.grad[...] = [3.0, 4.0]

    original_data = tensor.data
    original_grad = tensor.grad
    original_children = tensor.children
    original_ctx = tensor.ctx
    original_op = tensor.op
    original_label = tensor.label
    original_requires_grad = tensor.requires_grad

    kaiming_normal_(tensor, fan_in=2, nonlinearity="sigmoid")

    assert tensor.data is original_data
    assert tensor.grad is original_grad
    np.testing.assert_array_equal(tensor.grad, [3.0, 4.0])
    assert tensor.children is original_children
    assert tensor.ctx is original_ctx
    assert tensor.op == original_op
    assert tensor.label == original_label
    assert tensor.requires_grad is original_requires_grad
