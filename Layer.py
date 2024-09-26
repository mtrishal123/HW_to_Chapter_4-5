from Neuron import Neuron

class Layer:
    def __init__(self, num_neurons, input_size):
        # Initialize a layer with 'num_neurons' neurons, each receiving 'input_size' inputs
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]
        self.outputs = []

    def forward(self, inputs):
        # Perform forward pass for each neuron in the layer
        self.outputs = [neuron.forward(inputs) for neuron in self.neurons]
        return self.outputs
