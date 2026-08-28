import numpy as np
import torch

from baby_pytorch.nn import Tanh
from baby_pytorch.tensor import Tensor


def test_tanh_layer_has_no_parameters_or_weights():
    layer = Tanh()

    assert layer.parameters() == []
    assert layer.save_weights() == []
    assert layer.load_weights([]) is None
    assert repr(layer) == "Tanh"


def test_tanh_layer_values_and_gradients_match_pytorch():
    input_data = np.array([[-2.0, -0.5, 0.0], [0.25, 1.0, 2.5]])
    coefficient_data = np.array([[1.0, -2.0, 0.5], [3.0, 0.25, -1.5]])

    tensor = Tensor(input_data, requires_grad=True)
    coefficients = Tensor(coefficient_data)
    layer = Tanh()

    baby_result = layer(tensor)
    (baby_result * coefficients).sum().backward()

    torch_input = torch.tensor(
        input_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_result = torch.tanh(torch_input)
    (torch_result * torch.tensor(coefficient_data)).sum().backward()

    np.testing.assert_allclose(baby_result.data, torch_result.detach().numpy())
    np.testing.assert_allclose(tensor.grad, torch_input.grad.numpy())
    assert layer.out is baby_result


def test_tanh_layer_detaches_its_output_during_inference():
    tensor = Tensor([[-1.0, 0.0, 1.0]], requires_grad=True)

    result = Tanh()(tensor, training=False)

    np.testing.assert_allclose(result.data, np.tanh(tensor.data))
    assert not result.requires_grad
    assert result.children == []
