import numpy as np

class Activation:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        # Derivative of the sigmoid function
        sig = Activation.sigmoid(z)
        return sig * (1 - sig)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return np.where(z > 0, 1, 0)
