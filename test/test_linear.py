import numpy as np
import pytest
import torch

from baby_pytorch.nn import Linear
from baby_pytorch.optim import SGD
from baby_pytorch.tensor import Tensor


def test_linear_initializes_trainable_weights_and_bias():
    linear = Linear(2, 3)

    assert linear.weights.shape == (2, 3)
    assert linear.bias.shape == (3,)
    assert linear.weights.requires_grad
    assert linear.bias.requires_grad
    assert linear.parameters() == [linear.weights, linear.bias]
    assert repr(linear) == "Linear: weights: (2, 3) bias: (3,)"


def test_linear_can_omit_bias():
    linear = Linear(2, 3, bias=False)

    assert linear.bias is None
    assert linear.parameters() == [linear.weights]
    assert repr(linear) == "Linear: weights: (2, 3) bias: None"


def test_linear_without_bias_only_performs_matrix_multiplication():
    linear = Linear(2, 2, bias=False)
    linear.weights = Tensor([[1, 2], [3, 4]], requires_grad=True)
    tensor = Tensor([[2, 1], [-1, 3]])

    result = linear(tensor)

    np.testing.assert_array_equal(result.data, [[5, 8], [8, 10]])


def test_linear_values_and_gradients_match_pytorch():
    input_data = np.array([[0.2, -0.4, 1.0], [1.5, 0.3, -0.7]])
    weight_data = np.array(
        [[0.1, 0.5], [-0.2, 0.8], [1.2, -0.6]],
    )
    bias_data = np.array([0.25, -0.75])
    coefficient_data = np.array([[1.0, -2.0], [0.5, 3.0]])

    linear = Linear(3, 2)
    linear.weights = Tensor(weight_data, requires_grad=True)
    linear.bias = Tensor(bias_data, requires_grad=True)
    tensor = Tensor(input_data, requires_grad=True)
    coefficients = Tensor(coefficient_data)

    baby_result = linear(tensor)
    (baby_result * coefficients).sum().backward()

    torch_input = torch.tensor(
        input_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_weights = torch.tensor(
        weight_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_bias = torch.tensor(
        bias_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_result = torch_input @ torch_weights + torch_bias
    (torch_result * torch.tensor(coefficient_data)).sum().backward()

    np.testing.assert_allclose(baby_result.data, torch_result.detach().numpy())
    np.testing.assert_allclose(tensor.grad, torch_input.grad.numpy())
    np.testing.assert_allclose(linear.weights.grad, torch_weights.grad.numpy())
    np.testing.assert_allclose(linear.bias.grad, torch_bias.grad.numpy())


def test_linear_save_and_load_use_independent_tensor_leaves():
    linear = Linear(2, 3)
    linear.weights.data[:] = np.arange(6).reshape(2, 3)
    linear.bias.data[:] = [1, 2, 3]

    saved = linear.save_weights()
    linear.weights.data[0, 0] = 99
    linear.bias.data[0] = 99

    assert all(not parameter.requires_grad for parameter in saved)
    assert all(parameter.children == [] for parameter in saved)
    assert saved[0].data[0, 0] == 0
    assert saved[1].data[0] == 1

    restored = Linear(2, 3)
    optimizer = SGD(restored.parameters(), lr=0.1)
    original_parameters = restored.parameters()
    restored.load_weights(saved)

    np.testing.assert_array_equal(restored.weights.data, saved[0].data)
    np.testing.assert_array_equal(restored.bias.data, saved[1].data)
    assert all(parameter.requires_grad for parameter in restored.parameters())
    assert all(parameter.children == [] for parameter in restored.parameters())
    assert not np.shares_memory(restored.weights.data, saved[0].data)
    assert not np.shares_memory(restored.bias.data, saved[1].data)
    assert all(
        loaded is original
        for loaded, original in zip(restored.parameters(), original_parameters)
    )
    assert all(
        optimized is loaded
        for optimized, loaded in zip(optimizer.parameters, restored.parameters())
    )

    optimizer.zero_grad()
    restored(Tensor([[1.0, 2.0]])).sum().backward()
    weights_before_step = restored.weights.data.copy()
    optimizer.step()
    assert np.any(restored.weights.data != weights_before_step)


def test_linear_load_without_bias_keeps_bias_disabled():
    source = Linear(2, 3, bias=False)
    restored = Linear(2, 3, bias=False)

    restored.load_weights(source.save_weights())

    assert restored.bias is None
    assert restored.parameters() == [restored.weights]


def test_linear_load_rejects_a_different_bias_configuration():
    source = Linear(2, 3, bias=False)
    restored = Linear(2, 3)

    with pytest.raises(ValueError, match="Expected 2 weights, received 1"):
        restored.load_weights(source.save_weights())
