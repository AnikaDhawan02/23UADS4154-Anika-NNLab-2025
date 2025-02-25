# README

## Objective
WAP to implement a three-layer neural network using TensorFlow library (only, no Keras) to classify the MNIST handwritten digits dataset. Demonstrate the implementation of feed-forward and back-propagation approaches.

## Description of the Model
The model is a three-layer fully connected neural network designed for digit classification. It consists of:
- An input layer with 784 neurons (28x28 pixels flattened)
- Two hidden layers with 128 and 64 neurons, respectively
- An output layer with 10 neurons (one for each digit)
- The activation function used is **Sigmoid** 

### Code Implementation
```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0                                                                              
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

learning_rate = 0.01
num_epochs = 100
batch_size = 100
n_input = 784    # Input size (28x28)
n_hidden1 = 128 
n_hidden2 = 64   
n_output = 10   

initializer = tf.initializers.GlorotUniform()
W1 = tf.Variable(initializer([n_input, n_hidden1]))
b1 = tf.Variable(tf.zeros([n_hidden1]))

W2 = tf.Variable(initializer([n_hidden1, n_hidden2]))
b2 = tf.Variable(tf.zeros([n_hidden2]))

W3 = tf.Variable(initializer([n_hidden2, n_output]))
b3 = tf.Variable(tf.zeros([n_output]))

def forward_pass(x):
    z1 = tf.add(tf.matmul(x, W1), b1)
    a1 = tf.nn.sigmoid(z1)  

    z2 = tf.add(tf.matmul(a1, W2), b2)
    a2 = tf.nn.sigmoid(z2)  

    logits = tf.add(tf.matmul(a2, W3), b3)
    return logits

# Loss function
def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def compute_accuracy(logits, labels):
    correct_preds = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    return tf.reduce_mean(tf.cast(correct_preds, tf.float32))

optimizer = tf.optimizers.SGD(learning_rate)

for epoch in range(num_epochs):
    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        with tf.GradientTape() as tape:
            logits = forward_pass(x_batch)
            loss = compute_loss(logits, y_batch)

        gradients = tape.gradient(loss, [W1, b1, W2, b2, W3, b3])
        optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2, W3, b3]))

    test_logits = forward_pass(x_test)
    test_accuracy = compute_accuracy(test_logits, y_test)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.numpy():.4f}, Test Accuracy: {test_accuracy.numpy():.4f}")

final_accuracy = compute_accuracy(forward_pass(x_test), y_test)
print(f"Final Test Accuracy: {final_accuracy.numpy():.4f}")

```

## Description of the Code
1. The MNIST dataset is loaded and preprocessed.
2. The data is reshaped and normalized.
3. Labels are one-hot encoded.
4. Weights and biases are initialized using Glorot uniform initialization.
5. The forward pass is implemented using matrix operations and the Sigmoid activation function.
6. The loss is computed using softmax cross-entropy.
7. The optimizer updates the parameters using backpropagation.
8. The model is trained over multiple epochs with mini-batch gradient descent.
9. Accuracy is computed after each epoch.

## Training and Test Accuracy
```
Epoch 1/20, Loss: 2.2576, Test Accuracy: 0.3439
Epoch 2/20, Loss: 2.1871, Test Accuracy: 0.4653
Epoch 3/20, Loss: 2.0745, Test Accuracy: 0.5380
...
Epoch 20/20, Loss: 0.5091, Test Accuracy: 0.8763
Final Test Accuracy: 0.8763

Epoch 1/100, Loss: 2.2504, Test Accuracy: 0.3498
Epoch 2/100, Loss: 2.1721, Test Accuracy: 0.5024
Epoch 3/100, Loss: 2.0452, Test Accuracy: 0.5800
Epoch 4/100, Loss: 1.8516, Test Accuracy: 0.6409
...
Epoch 100/100, Loss: 0.2554, Test Accuracy: 0.9333
Final Test Accuracy: 0.9333
```

## Performance Evaluation
The model successfully classifies handwritten digits with a final test accuracy of **93.33%**. The loss continuously decreases, showing good convergence.

## Conclusion
The implementation of a three-layer neural network using **only TensorFlow (no Keras)** effectively classifies MNIST digits. The feed-forward and backpropagation approaches are demonstrated, leading to a well-trained model. The choice of the **Sigmoid activation function** ensures smooth gradient updates, though ReLU might improve convergence speed. The results validate the correctness and efficiency of the model. Further improvements could include hyperparameter tuning or switching to a deeper architecture for enhanced performance.
