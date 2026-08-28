import numpy as np
import torch

from baby_pytorch.nn import Relu
from baby_pytorch.tensor import Tensor


def test_relu_layer_has_no_parameters_or_weights():
    layer = Relu()

    assert layer.parameters() == []
    assert layer.save_weights() == []
    assert layer.load_weights([]) is None
    assert repr(layer) == "Relu"


def test_relu_layer_values_and_gradients_match_pytorch():
    input_data = np.array([[-2.0, -0.5, 0.0], [0.25, 1.0, 2.5]])
    coefficient_data = np.array([[1.0, -2.0, 0.5], [3.0, 0.25, -1.5]])
    tensor = Tensor(input_data, requires_grad=True)
    coefficients = Tensor(coefficient_data)
    layer = Relu()

    baby_result = layer(tensor)
    (baby_result * coefficients).sum().backward()

    torch_input = torch.tensor(
        input_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_result = torch.relu(torch_input)
    (torch_result * torch.tensor(coefficient_data)).sum().backward()

    np.testing.assert_allclose(baby_result.data, torch_result.detach().numpy())
    np.testing.assert_allclose(tensor.grad, torch_input.grad.numpy())
    assert layer.out is baby_result


def test_relu_layer_detaches_its_output_during_inference():
    tensor = Tensor([[-1.0, 0.0, 1.0]], requires_grad=True)
    layer = Relu()
    layer.eval()

    result = layer(tensor)

    np.testing.assert_array_equal(result.data, [[0.0, 0.0, 1.0]])
    assert not result.requires_grad
    assert result.children == []
