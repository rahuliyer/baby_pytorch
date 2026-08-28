import numpy as np
import torch

from baby_pytorch.nn import BatchNorm1d
from baby_pytorch.tensor import Tensor


def test_batch_norm_initializes_parameters_and_running_statistics():
    batch_norm = BatchNorm1d(3)

    np.testing.assert_array_equal(batch_norm.gamma.data, np.ones(3))
    np.testing.assert_array_equal(batch_norm.beta.data, np.zeros(3))
    np.testing.assert_array_equal(batch_norm.running_mean.data, np.zeros(3))
    np.testing.assert_array_equal(batch_norm.running_var.data, np.ones(3))
    assert batch_norm.gamma.requires_grad
    assert batch_norm.beta.requires_grad
    assert not batch_norm.running_mean.requires_grad
    assert not batch_norm.running_var.requires_grad
    assert batch_norm.parameters() == [batch_norm.gamma, batch_norm.beta]
    assert repr(batch_norm) == (
        "BatchNorm1d: self.gamma.shape=(3,), self.beta.shape=(3,)"
    )


def test_batch_norm_training_values_and_gradients_match_notebook_formula():
    input_data = np.array(
        [
            [0.2, -0.5, 1.0],
            [1.2, 0.3, -0.7],
            [-0.4, 1.5, 0.2],
            [0.8, -1.0, 1.7],
        ],
    )
    gamma_data = np.array([1.2, -0.7, 0.5])
    beta_data = np.array([0.1, 0.3, -0.2])
    coefficient_data = np.array(
        [
            [1.0, -2.0, 0.5],
            [0.5, 3.0, -1.0],
            [-0.75, 0.25, 2.0],
            [1.5, -0.5, 0.8],
        ],
    )

    batch_norm = BatchNorm1d(3, eps=1e-5, momentum=0.2)
    batch_norm.gamma = Tensor(gamma_data, requires_grad=True)
    batch_norm.beta = Tensor(beta_data, requires_grad=True)
    tensor = Tensor(input_data, requires_grad=True)

    baby_result = batch_norm(tensor)
    (baby_result * Tensor(coefficient_data)).sum().backward()

    torch_input = torch.tensor(
        input_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_gamma = torch.tensor(
        gamma_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_beta = torch.tensor(
        beta_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_mean = torch_input.mean(dim=0)
    torch_var = torch_input.var(dim=0)
    torch_result = (
        (torch_input - torch_mean) / (torch_var + 1e-5) ** 0.5
        * torch_gamma
        + torch_beta
    )
    (torch_result * torch.tensor(coefficient_data)).sum().backward()

    np.testing.assert_allclose(
        baby_result.data,
        torch_result.detach().numpy(),
        rtol=1e-7,
        atol=1e-9,
    )
    np.testing.assert_allclose(tensor.grad, torch_input.grad.numpy())
    np.testing.assert_allclose(batch_norm.gamma.grad, torch_gamma.grad.numpy())
    np.testing.assert_allclose(batch_norm.beta.grad, torch_beta.grad.numpy())


def test_batch_norm_updates_detached_running_statistics():
    input_data = np.array(
        [[1.0, 3.0], [2.0, 7.0], [6.0, 11.0], [3.0, 15.0]],
    )
    batch_norm = BatchNorm1d(2, momentum=0.25)

    batch_norm(Tensor(input_data, requires_grad=True))

    expected_mean = 0.75 * np.zeros(2) + 0.25 * input_data.mean(axis=0)
    expected_var = 0.75 * np.ones(2) + 0.25 * input_data.var(
        axis=0,
        ddof=1,
    )
    np.testing.assert_allclose(batch_norm.running_mean.data, expected_mean)
    np.testing.assert_allclose(batch_norm.running_var.data, expected_var)
    assert not batch_norm.running_mean.requires_grad
    assert not batch_norm.running_var.requires_grad
    assert batch_norm.running_mean.children == []
    assert batch_norm.running_var.children == []


def test_batch_norm_evaluation_uses_frozen_running_statistics():
    batch_norm = BatchNorm1d(2, eps=0.01)
    batch_norm.gamma.data[:] = [2.0, 0.5]
    batch_norm.beta.data[:] = [-1.0, 3.0]
    batch_norm.running_mean.data[:] = [1.0, -2.0]
    batch_norm.running_var.data[:] = [4.0, 9.0]
    tensor = Tensor([[3.0, 1.0], [-1.0, -5.0]], requires_grad=True)
    original_mean = batch_norm.running_mean.data.copy()
    original_var = batch_norm.running_var.data.copy()

    result = batch_norm(tensor, training=False)

    expected = (
        (tensor.data - original_mean) / np.sqrt(original_var + 0.01)
        * batch_norm.gamma.data
        + batch_norm.beta.data
    )
    np.testing.assert_allclose(result.data, expected)
    np.testing.assert_array_equal(batch_norm.running_mean.data, original_mean)
    np.testing.assert_array_equal(batch_norm.running_var.data, original_var)
    assert not result.requires_grad
    assert result.children == []


def test_batch_norm_save_and_load_copy_parameters_and_running_statistics():
    batch_norm = BatchNorm1d(2)
    batch_norm.gamma.data[:] = [1.5, 0.5]
    batch_norm.beta.data[:] = [-0.5, 2.0]
    batch_norm.running_mean.data[:] = [3.0, 4.0]
    batch_norm.running_var.data[:] = [5.0, 6.0]

    saved = batch_norm.save_weights()
    batch_norm.gamma.data[0] = 99
    batch_norm.running_mean.data[0] = 99

    assert len(saved) == 4
    assert all(not tensor.requires_grad for tensor in saved)
    assert all(tensor.children == [] for tensor in saved)
    assert saved[0].data[0] == 1.5
    assert saved[2].data[0] == 3.0

    restored = BatchNorm1d(2)
    restored.load_weights(saved)

    for actual, expected in zip(restored.save_weights(), saved):
        np.testing.assert_array_equal(actual.data, expected.data)
        assert not np.shares_memory(actual.data, expected.data)
    assert restored.gamma.requires_grad
    assert restored.beta.requires_grad
    assert not restored.running_mean.requires_grad
    assert not restored.running_var.requires_grad
