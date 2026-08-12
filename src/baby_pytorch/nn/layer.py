from baby_pytorch.nn import Neuron

class Layer:
    def __init__(self, in_size, out_size):
        self.neurons = [Neuron(in_size) for _ in range(out_size)]

    def __call__(self, input):
        return [neuron(input) for neuron in self.neurons]

    def parameters(self):
        parameters = []

        for neuron in self.neurons:
            parameters += neuron.parameters()

        return parameters