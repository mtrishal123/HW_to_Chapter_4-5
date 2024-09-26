from ForwardProp import ForwardProp
from BackProp import BackProp


class GradDescent:
    def __init__(self, model, loss_function, learning_rate=0.01):
        self.model = model
        self.forward_prop = ForwardProp(model)
        self.back_prop = BackProp(model, loss_function)
        self.learning_rate = learning_rate
    
    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            predictions = self.forward_prop.forward(inputs)
            loss = self.back_prop.loss_function.mse(predictions, targets)
            self.back_prop.backward(inputs, targets, predictions, self.learning_rate)
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}')
