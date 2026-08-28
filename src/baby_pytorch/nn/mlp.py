from baby_pytorch.nn.linear import Linear
from baby_pytorch.nn.module import Module


class MLP(Module):
    def __init__(self, input_size, hidden_layers, out_size, activation):
        super().__init__()
        if isinstance(activation, Module) and activation.parameters():
            raise ValueError("MLP activations must be parameter-free.")

        layer_sizes = [input_size, *hidden_layers, out_size]
        self.layers = [
            Linear(fan_in, fan_out)
            for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:])
        ]
        self.activation = activation

    def forward(self, x, training):
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(self.activation, Module):
                x = self.activation(x)
            else:
                x = self.activation(x)

        # The final layer produces logits, so it has no activation.
        return self.layers[-1](x)

    def parameters(self):
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]

    def save_weights(self):
        return [
            weight
            for layer in self.layers
            for weight in layer.save_weights()
        ]

    def load_weights(self, weights):
        expected_count = sum(
            len(layer.save_weights()) for layer in self.layers
        )
        if len(weights) != expected_count:
            raise ValueError(
                f"Expected {expected_count} weights, received {len(weights)}."
            )

        offset = 0
        for layer in self.layers:
            weight_count = len(layer.save_weights())
            layer.load_weights(weights[offset:offset + weight_count])
            offset += weight_count

    def __repr__(self):
        activation_name = (
            repr(self.activation)
            if isinstance(self.activation, Module)
            else getattr(self.activation, "__name__", repr(self.activation))
        )
        layers = "\n".join(
            f"  ({index}): {layer!r}" for index, layer in enumerate(self.layers)
        )
        return f"{self.__class__.__name__}(\n{layers}\n  activation: {activation_name}\n)"
