from baby_pytorch.nn import MLP
from baby_pytorch.activation_functions import relu

def test_empty_hidden_layers():
    input_size = 3
    hidden_layers = []
    out_size = 2
    activation = relu

    mlp = MLP(input_size, hidden_layers, out_size, activation)

    assert len(mlp.layers) == 1  # Only the output layer should be present

def test_mlp_parameters():
    input_size = 3
    hidden_layers = [5, 4]
    out_size = 2
    activation = relu

    mlp = MLP(input_size, hidden_layers, out_size, activation)

    layers = [input_size] + hidden_layers + [out_size]
    # Add 1 for the bias term for each layer

    expected_param_count = sum(
        (layers[i] + 1) * layers[i + 1] for i in range(len(layers) - 1)
    )

    assert len(mlp.parameters()) == expected_param_count