import numpy as np

class Neuron:
    def __init__(self, input_size):
        # Initialize weights to match the number of input features (input_size)
        self.weights = np.random.randn(input_size)  # Correct shape to match input size (100 in this case)
        self.bias = np.random.randn()  # Bias remains a scalar

    def forward(self, inputs):
        # Perform the weighted sum (z = Wx + b)
        print(f"Input shape: {inputs.shape}")
        print(f"Weight shape: {self.weights.shape}")
        self.z = np.dot(inputs, self.weights) + self.bias
        return self.z
