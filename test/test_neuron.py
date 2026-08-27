from baby_pytorch.nn import Neuron
from baby_pytorch.tensor import Tensor

import pytest

def test_neuron_size_match():
    neuron = Neuron(in_size=3)
    input = [1, 1]

    with pytest.raises(ValueError):
        neuron(input)

def test_neuron_output():
    neuron = Neuron(in_size=3)
    neuron.weights = [
        Tensor(0.5, requires_grad=True),
        Tensor(-0.5, requires_grad=True),
        Tensor(1.0, requires_grad=True)
    ]
    neuron.bias = Tensor(0.25, requires_grad=True)
    input = [1, 2, 3]

    output = neuron(input)

    assert output.data == 2.75

def test_neuron_parameters():
    neuron = Neuron(in_size=3)

    assert len(neuron.parameters()) == 4  # 3 weights + 1 bias