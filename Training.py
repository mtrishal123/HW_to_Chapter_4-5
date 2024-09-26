from GradDescent import GradDescent

class Training:
    def __init__(self, model, loss_function, learning_rate=0.01):
        self.optimizer = GradDescent(model, loss_function, learning_rate)
    
    def run(self, X_train, y_train, epochs):
        self.optimizer.train(X_train, y_train, epochs)
