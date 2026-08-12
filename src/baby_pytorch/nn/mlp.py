from baby_pytorch.nn import Layer

class MLP:
    def __init__(self, input_size, hidden_layers, out_size, activation):
        self.layers = []
        self.activation = activation

        in_size = input_size
        for i in range(len(hidden_layers)):
            self.layers.append(Layer(in_size, hidden_layers[i]))
            in_size = hidden_layers[i]

        self.layers.append(Layer(in_size, out_size))

    def __call__(self, input):
        for i in range(len(self.layers) - 1):
            input = [self.activation(output) for output in self.layers[i](input)]

        # No activation for the output layer
        return self.layers[-1](input)

    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters += layer.parameters()

        return parameters