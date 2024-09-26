import numpy as np

class Parameters:
    def __init__(self, input_size, hidden_size, output_size):
        # Weight and bias initialization for input -> hidden layer
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.random.randn(hidden_size)
        
        # Weight and bias initialization for hidden -> output layer
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.random.randn(output_size)
