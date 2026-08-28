import numpy as np
import torch

from baby_pytorch.tensor import Tensor


def _build_complex_graph(
    x,
    weight,
    bias,
    scale,
    column_scale,
    coefficients,
    element_weights,
    row_weights,
    statistics,
    row_mean,
    transpose,
):
    projected = x @ weight + bias
    activated = (projected**2 + 1.0).log()
    selected = activated[:, [0, 2, 2, 4]]
    scaled = selected * scale * column_scale
    smooth = (scaled.exp() + 2.0).log()
    statistics_value = statistics(scaled)
    weighted_smooth = smooth * element_weights
    row_summary = row_mean(scaled)
    layout_branch = transpose(selected).reshape(2, 6)
    column_loss = (
        (weighted_smooth.mean(dim=0) + statistics_value) * coefficients
    ).sum()
    row_loss = (row_summary * row_weights).sum()
    layout_loss = (layout_branch**2).mean()
    loss = column_loss + 0.25 * row_loss + 0.1 * layout_loss

    return {
        "projected": projected,
        "activated": activated,
        "selected": selected,
        "scaled": scaled,
        "smooth": smooth,
        "statistics": statistics_value,
        "weighted_smooth": weighted_smooth,
        "row_summary": row_summary,
        "layout_branch": layout_branch,
        "loss": loss,
    }


def test_complex_graph_matches_pytorch_values_and_gradients():
    rng = np.random.default_rng(42)
    x_data = rng.normal(scale=0.5, size=(3, 4))
    weight_data = rng.normal(scale=0.3, size=(4, 5))
    bias_data = rng.normal(scale=0.1, size=(1, 5))
    scale_data = np.array([0.7, -1.1, 0.5, 0.9])
    column_scale_data = np.array([[0.8], [1.2], [-0.6]])
    coefficient_data = np.array([1.5, -0.75, 2.0, 0.4])
    element_weight_data = rng.normal(scale=0.7, size=(3, 4))
    row_weight_data = np.array([[0.5], [-1.25], [2.0]])

    x = Tensor(x_data, requires_grad=True)
    weight = Tensor(weight_data, requires_grad=True)
    bias = Tensor(bias_data, requires_grad=True)
    scale = Tensor(scale_data, requires_grad=True)
    column_scale = Tensor(column_scale_data, requires_grad=True)
    coefficients = Tensor(coefficient_data)
    element_weights = Tensor(element_weight_data)
    row_weights = Tensor(row_weight_data)

    baby_graph = _build_complex_graph(
        x,
        weight,
        bias,
        scale,
        column_scale,
        coefficients,
        element_weights,
        row_weights,
        statistics=lambda value: value.var(dim=0) + value.std(dim=0),
        row_mean=lambda value: value.mean(dim=1, keepdims=True),
        transpose=lambda value: value.T(),
    )
    baby_graph["loss"].backward()

    torch_x = torch.tensor(x_data, dtype=torch.float64, requires_grad=True)
    torch_weight = torch.tensor(
        weight_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_bias = torch.tensor(
        bias_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_scale = torch.tensor(
        scale_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_column_scale = torch.tensor(
        column_scale_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_coefficients = torch.tensor(coefficient_data, dtype=torch.float64)
    torch_element_weights = torch.tensor(element_weight_data, dtype=torch.float64)
    torch_row_weights = torch.tensor(row_weight_data, dtype=torch.float64)

    torch_graph = _build_complex_graph(
        torch_x,
        torch_weight,
        torch_bias,
        torch_scale,
        torch_column_scale,
        torch_coefficients,
        torch_element_weights,
        torch_row_weights,
        statistics=lambda value: value.var(dim=0) + value.std(dim=0),
        row_mean=lambda value: value.mean(dim=1, keepdim=True),
        transpose=lambda value: value.T,
    )
    torch_graph["loss"].backward()

    for name, baby_value in baby_graph.items():
        np.testing.assert_allclose(
            baby_value.data,
            torch_graph[name].detach().numpy(),
            rtol=1e-7,
            atol=1e-9,
        )

    for baby_tensor, torch_tensor in [
        (x, torch_x),
        (weight, torch_weight),
        (bias, torch_bias),
        (scale, torch_scale),
        (column_scale, torch_column_scale),
    ]:
        np.testing.assert_allclose(
            baby_tensor.grad,
            torch_tensor.grad.numpy(),
            rtol=1e-6,
            atol=1e-8,
        )
        assert baby_tensor.grad.shape == tuple(torch_tensor.grad.shape)
