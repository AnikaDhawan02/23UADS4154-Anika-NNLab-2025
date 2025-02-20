import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.randn(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias term
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias column
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.activation(np.dot(self.weights, X[i]))
                self.weights += self.learning_rate * (y[i] - y_pred) * X[i]

    def evaluate(self, X, y):
        y_pred = np.array([self.predict(x) for x in X])
        accuracy = np.mean(y_pred == y) * 100
        return accuracy, y_pred

def train_perceptron(X, y, name):
    p = Perceptron(input_size=X.shape[1])
    p.train(X, y)
    accuracy, predictions = p.evaluate(X, y)
    print(f"{name} Accuracy: {accuracy:.2f}% | Predictions: {predictions}")
    return predictions, y

fun_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

fun1_y = np.array([0, 0, 0, 1])  # NAND
fun2_y = np.array([0, 0, 1, 0])  # Custom function
fun3_y = np.array([0, 1, 0, 0])  # Custom function
fun4_y = np.array([1, 0, 0, 0])  # Custom function

fun1_predictions, _ = train_perceptron(fun_X, fun1_y, "Fun1")
fun2_predictions, _ = train_perceptron(fun_X, fun2_y, "Fun2")
fun3_predictions, _ = train_perceptron(fun_X, fun3_y, "Fun3")
fun4_predictions, _ = train_perceptron(fun_X, fun4_y, "Fun4")

final_X = np.column_stack([fun1_predictions, fun2_predictions, fun3_predictions, fun4_predictions])
final_y = np.array([0, 1, 1, 0])

final_predictions, actual_y = train_perceptron(final_X, final_y, "Final Perceptron")
