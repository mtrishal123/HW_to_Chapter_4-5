import numpy as np

class LossFunction:
    @staticmethod
    def mse(predictions, targets):
        return np.mean(np.power(targets - predictions, 2))
    
    @staticmethod
    def mse_derivative(predictions, targets):
        return 2 * (predictions - targets) / targets.size
