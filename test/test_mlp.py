import numpy as np

from baby_pytorch.activation_functions import tanh
from baby_pytorch.nn import BatchNorm1d, Linear, MLP, Module, Tanh
from baby_pytorch.tensor import Tensor


def test_mlp_is_a_module_with_array_backed_linear_layers():
    mlp = MLP(3, [5, 4], 2, tanh)

    assert isinstance(mlp, Module)
    assert all(isinstance(layer, Linear) for layer in mlp.layers)
    assert [layer.weights.shape for layer in mlp.layers] == [
        (3, 5),
        (5, 4),
        (4, 2),
    ]
    assert [layer.bias.shape for layer in mlp.layers] == [(5,), (4,), (2,)]


def test_mlp_batched_forward_has_expected_shape():
    mlp = MLP(3, [5, 4], 2, tanh)

    result = mlp(Tensor(np.ones((7, 3))))

    assert result.shape == (7, 2)


def test_mlp_parameters_are_flattened_in_layer_order():
    mlp = MLP(3, [5, 4], 2, tanh)

    expected = [
        parameter
        for layer in mlp.layers
        for parameter in layer.parameters()
    ]

    assert mlp.parameters() == expected
    assert len(mlp.parameters()) == 6


def test_empty_hidden_layers_only_apply_the_output_linear_layer():
    def activation_that_must_not_run(_):
        raise AssertionError("The output layer must not have an activation")

    mlp = MLP(3, [], 2, activation_that_must_not_run)
    mlp.layers[0].weights = Tensor(
        [[1, 0], [0, 1], [1, -1]],
        requires_grad=True,
    )
    mlp.layers[0].bias = Tensor([0.5, -0.5], requires_grad=True)

    result = mlp(Tensor([[1, 2, 3], [-1, 0, 2]]))

    np.testing.assert_array_equal(result.data, [[4.5, -1.5], [1.5, -2.5]])
    assert len(mlp.layers) == 1


def test_mlp_supports_functional_and_module_activations():
    functional_mlp = MLP(2, [2], 1, tanh)
    module_mlp = MLP(2, [2], 1, Tanh())

    for mlp in (functional_mlp, module_mlp):
        mlp.layers[0].weights = Tensor(np.eye(2), requires_grad=True)
        mlp.layers[0].bias = Tensor(np.zeros(2), requires_grad=True)
        mlp.layers[1].weights = Tensor([[1], [-2]], requires_grad=True)
        mlp.layers[1].bias = Tensor([0.25], requires_grad=True)

    inputs = Tensor([[0.5, -1.0], [1.5, 0.25]])
    expected = np.tanh(inputs.data) @ np.array([[1], [-2]]) + 0.25

    np.testing.assert_allclose(functional_mlp(inputs).data, expected)
    np.testing.assert_allclose(module_mlp(inputs).data, expected)


def test_mlp_backward_populates_shaped_parameter_gradients():
    mlp = MLP(2, [3], 1, Tanh())
    inputs = Tensor([[0.2, -0.4], [1.0, 0.5]], requires_grad=True)

    mlp(inputs).sum().backward()

    assert inputs.grad.shape == inputs.shape
    assert np.any(inputs.grad != 0)
    for parameter in mlp.parameters():
        assert parameter.grad.shape == parameter.shape
        assert np.all(np.isfinite(parameter.grad))
        assert np.any(parameter.grad != 0)


def test_mlp_inference_returns_a_detached_tensor():
    mlp = MLP(2, [3], 1, Tanh())
    inputs = Tensor([[0.2, -0.4], [1.0, 0.5]], requires_grad=True)

    result = mlp(inputs, training=False)

    assert result.shape == (2, 1)
    assert not result.requires_grad
    assert result.children == []
    result.sum().backward()
    np.testing.assert_array_equal(inputs.grad, np.zeros_like(inputs.data))
    for parameter in mlp.parameters():
        np.testing.assert_array_equal(parameter.grad, np.zeros(parameter.shape))


def test_mlp_save_and_load_use_independent_tensor_leaves():
    mlp = MLP(2, [3], 1, Tanh())
    for index, parameter in enumerate(mlp.parameters()):
        parameter.data.fill(index + 1)

    saved = mlp.save_weights()
    mlp.parameters()[0].data.fill(99)

    assert len(saved) == 4
    assert all(not weight.requires_grad for weight in saved)
    assert all(weight.children == [] for weight in saved)
    assert np.all(saved[0].data == 1)

    restored = MLP(2, [3], 1, Tanh())
    restored.load_weights(saved)

    for parameter, weight in zip(restored.parameters(), saved):
        np.testing.assert_array_equal(parameter.data, weight.data)
        assert parameter.requires_grad
        assert parameter.children == []
        assert not np.shares_memory(parameter.data, weight.data)


def test_mlp_includes_module_activation_parameters_and_state():
    activation = BatchNorm1d(2)
    mlp = MLP(2, [2], 1, activation)
    activation.gamma.data[:] = [1.5, 0.5]
    activation.beta.data[:] = [-0.5, 2.0]
    activation.running_mean.data[:] = [3.0, 4.0]
    activation.running_var.data[:] = [5.0, 6.0]

    assert mlp.parameters() == [
        *mlp.layers[0].parameters(),
        *mlp.layers[1].parameters(),
        activation.gamma,
        activation.beta,
    ]

    saved = mlp.save_weights()

    assert len(saved) == 8
    restored = MLP(2, [2], 1, BatchNorm1d(2))
    original_parameters = restored.parameters()
    restored.load_weights(saved)

    assert all(
        loaded is original
        for loaded, original in zip(restored.parameters(), original_parameters)
    )
    for actual, expected in zip(restored.save_weights(), saved):
        np.testing.assert_array_equal(actual.data, expected.data)


def test_mlp_load_rejects_the_wrong_number_of_weights():
    mlp = MLP(2, [3], 1, Tanh())

    try:
        mlp.load_weights(mlp.save_weights()[:-1])
    except ValueError as error:
        assert str(error) == "Expected 4 weights, received 3."
    else:
        raise AssertionError("Expected load_weights to reject incomplete weights")


def test_mlp_representation_describes_layers_and_activation():
    mlp = MLP(2, [3], 1, Tanh())

    assert repr(mlp) == (
        "MLP(\n"
        "  (0): Linear: weights: (2, 3) bias: (3,)\n"
        "  (1): Linear: weights: (3, 1) bias: (1,)\n"
        "  activation: Tanh\n"
        ")"
    )
