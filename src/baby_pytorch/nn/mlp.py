from baby_pytorch.nn.linear import Linear
from baby_pytorch.nn.module import Module


class MLP(Module):
    def __init__(self, input_size, hidden_layers, out_size, activation):
        layer_sizes = [input_size, *hidden_layers, out_size]
        self.layers = [
            Linear(fan_in, fan_out)
            for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:])
        ]
        self.activation = activation

    def forward(self, x, training):
        for layer in self.layers[:-1]:
            x = layer(x, training=training)
            if isinstance(self.activation, Module):
                x = self.activation(x, training=training)
            else:
                x = self.activation(x)

        # The final layer produces logits, so it has no activation.
        return self.layers[-1](x, training=training)

    def parameters(self):
        parameters = [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]
        if isinstance(self.activation, Module):
            parameters += self.activation.parameters()
        return parameters

    def save_weights(self):
        weights = [
            weight
            for layer in self.layers
            for weight in layer.save_weights()
        ]
        if isinstance(self.activation, Module):
            weights += self.activation.save_weights()
        return weights

    def load_weights(self, weights):
        linear_weight_count = sum(
            len(layer.save_weights()) for layer in self.layers
        )
        activation_weight_count = (
            len(self.activation.save_weights())
            if isinstance(self.activation, Module)
            else 0
        )
        expected_count = linear_weight_count + activation_weight_count
        if len(weights) != expected_count:
            raise ValueError(
                f"Expected {expected_count} weights, received {len(weights)}."
            )

        offset = 0
        for layer in self.layers:
            weight_count = len(layer.save_weights())
            layer.load_weights(weights[offset:offset + weight_count])
            offset += weight_count

        if isinstance(self.activation, Module):
            self.activation.load_weights(weights[offset:])

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
