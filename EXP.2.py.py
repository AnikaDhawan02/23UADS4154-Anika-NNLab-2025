
import numpy as np

def step_function(x):
    return np.where(x >= 0, 1, 0)

def step_derivative(x):
    return np.zeros_like(x)  # Zero derivative for the step function

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.random.randn(hidden_size)
        self.bias_output = np.random.randn(output_size)
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = step_function(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = step_function(self.final_input)
        return self.final_output
    
    def backward(self, X, y, output):
        error = y - output
        d_output = error * step_derivative(output)
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * step_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0) * self.learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0) * self.learning_rate

    def train(self, X, y):
        for _ in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X)

# XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Train MLP using step function
mlp = MLP(input_size=2, hidden_size=2, output_size=1)
mlp.train(X_xor, y_xor)
    
predictions = mlp.predict(X_xor)

# Tabled output for clarity
import pandas as pd
output_table = pd.DataFrame({
    'Input 1': X_xor[:, 0],
    'Input 2': X_xor[:, 1],
    'Expected Output': y_xor.flatten(),
    'Predicted Output': predictions.astype(int).flatten()
})

print("\nXOR MLP Predictions with Step Function (Table Format):\n")
print(output_table.to_string(index=False))

