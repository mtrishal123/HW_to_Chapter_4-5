from Activation import Activation
import numpy as np

class BackProp:
    def __init__(self, model, loss_function):
        self.model = model
        self.loss_function = loss_function
    
    def backward(self, inputs, targets, predictions, learning_rate):
        # Calculate output layer error
        output_error = self.loss_function.mse_derivative(predictions, targets) * Activation.sigmoid_derivative(predictions)
        
        # Backpropagate to hidden layer
        hidden_error = np.dot(output_error, self.model.params.weights_hidden_output.T) * Activation.relu_derivative(self.model.hidden_output)
        
        # Update weights and biases
        self.model.params.weights_hidden_output -= learning_rate * np.dot(self.model.hidden_output.T, output_error)
        self.model.params.bias_output -= learning_rate * np.sum(output_error, axis=0)
        
        self.model.params.weights_input_hidden -= learning_rate * np.dot(inputs.T, hidden_error)
        self.model.params.bias_hidden -= learning_rate * np.sum(hidden_error, axis=0)
