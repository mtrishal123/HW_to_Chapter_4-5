import numpy as np
from Parameters import Parameters

class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = Parameters(input_size, hidden_size, output_size)

    def forward(self, X):
        # Forward pass through the hidden layer
        hidden_input = np.dot(X, self.params.weights_input_hidden) + self.params.bias_hidden
        hidden_output = self.relu(hidden_input)
        
        # Forward pass through the output layer
        output_input = np.dot(hidden_output, self.params.weights_hidden_output) + self.params.bias_output
        output = self.sigmoid(output_input)
        
        self.hidden_output = hidden_output  # Store for backprop
        return output

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
