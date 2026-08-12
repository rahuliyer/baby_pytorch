from baby_pytorch.nn import Layer

def test_layer_parameters():
    in_size = 3
    out_size = 7

    layer = Layer(3, 7)

    expected_param_count = (in_size + 1) * out_size

    assert len(layer.parameters()) == expected_param_count