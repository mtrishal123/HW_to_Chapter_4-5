import numpy as np
from Model import Model
from LossFunction import LossFunction
from Training import Training

# Generate sample data (for demonstration purposes, replace with actual dataset)
def generate_data(samples, input_size):
    X = np.random.randn(samples, input_size)  # Random inputs
    y = np.random.randint(0, 2, (samples, 1))  # Binary targets (0 or 1)
    return X, y

def main():
    # Define network parameters
    input_size = 10  # Example input size
    hidden_size = 5  # Number of neurons in the hidden layer
    output_size = 1  # Example for binary classification
    learning_rate = 0.01
    epochs = 2000

    # Create a model with one hidden layer
    model = Model(input_size, hidden_size, output_size)
    
    # Create a loss function (Mean Squared Error in this case)
    loss_function = LossFunction()

    # Generate some example data
    X_train, y_train = generate_data(100, input_size)  # 100 samples

    # Initialize training
    trainer = Training(model, loss_function, learning_rate)

    # Train the model
    trainer.run(X_train, y_train, epochs)

    # Example prediction
    sample_input = np.random.randn(1, input_size)  # Example input
    prediction = model.forward(sample_input)
    print(f"Prediction for sample input: {prediction}")

if __name__ == '__main__':
    main()
