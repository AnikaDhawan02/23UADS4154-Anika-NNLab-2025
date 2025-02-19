import numpy as np

# Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.randn(input_size + 1) 
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        
        # Add bias input (1)
        inputs = np.insert(inputs, 0, 1)  
        return self.activation(np.dot(inputs, self.weights))

    def train(self, training_inputs, labels):
      
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                # Add bias input (1)
                inputs = np.insert(inputs, 0, 1) 
                prediction = self.predict(inputs[1:]) 
                error = label - prediction
                self.weights += self.learning_rate * error * inputs 

# NAND truth table
nand_inputs = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

nand_labels = np.array([1, 1, 1, 0])  # NAND truth table output

# XOR truth table
xor_inputs = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])

xor_labels = np.array([0, 1, 1, 0])  # XOR truth table output


print("Training on NAND data")
perceptron_nand = Perceptron(input_size=2, learning_rate=0.1, epochs=100)
perceptron_nand.train(nand_inputs, nand_labels)


print("Predictions for NAND:")
for inputs in nand_inputs:
    print(f"Input: {inputs}, Prediction: {perceptron_nand.predict(inputs)}")


print("\nTraining on XOR data")
perceptron_xor = Perceptron(input_size=2, learning_rate=0.1, epochs=100)
perceptron_xor.train(xor_inputs, xor_labels)


print("Predictions for XOR:")
for inputs in xor_inputs:
    print(f"Input: {inputs}, Prediction: {perceptron_xor.predict(inputs)}")
