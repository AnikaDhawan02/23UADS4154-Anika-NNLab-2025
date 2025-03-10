import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# Hyperparameter combinations
hidden_layer_configs = [(160, 100), (100, 100), (100, 160), (60, 60), (100, 60)]
learning_rates = [0.01, 0.1, 1]
num_epochs = 20
batch_size = 10
n_input = 784
n_output = 10

# Iterate over different configurations
for (n_hidden1, n_hidden2) in hidden_layer_configs:
    for learning_rate in learning_rates:
        print(f"\nTraining with Hidden Layers: ({n_hidden1}, {n_hidden2}), Learning Rate: {learning_rate}")
        
        # Initialize weights and biases
        initializer = tf.initializers.GlorotUniform()
        W1 = tf.Variable(initializer([n_input, n_hidden1]))
        b1 = tf.Variable(tf.zeros([n_hidden1]))
        W2 = tf.Variable(initializer([n_hidden1, n_hidden2]))
        b2 = tf.Variable(tf.zeros([n_hidden2]))
        W3 = tf.Variable(initializer([n_hidden2, n_output]))
        b3 = tf.Variable(tf.zeros([n_output]))

        # Define forward pass
        def forward_pass(x):
            z1 = tf.add(tf.matmul(x, W1), b1)
            a1 = tf.nn.relu(z1)  # ReLU activation
            z2 = tf.add(tf.matmul(a1, W2), b2)
            a2 = tf.nn.relu(z2)  # ReLU activation
            logits = tf.add(tf.matmul(a2, W3), b3)
            return logits

        # Loss function
        def compute_loss(logits, labels):
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        # Accuracy metric
        def compute_accuracy(logits, labels):
            correct_preds = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
            return tf.reduce_mean(tf.cast(correct_preds, tf.float32))

        # Optimizer
        optimizer = tf.optimizers.SGD(learning_rate)
        
        # Training loop
        train_losses, test_accuracies = [], []
        start_time = time.time()
        for epoch in range(num_epochs):
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                
                with tf.GradientTape() as tape:
                    logits = forward_pass(x_batch)
                    loss = compute_loss(logits, y_batch)
                
                gradients = tape.gradient(loss, [W1, b1, W2, b2, W3, b3])
                optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2, W3, b3]))
            
            # Evaluate after each epoch
            test_logits = forward_pass(x_test)
            test_accuracy = compute_accuracy(test_logits, y_test)
            train_losses.append(loss.numpy())
            test_accuracies.append(test_accuracy.numpy())
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.numpy():.4f}, Test Accuracy: {test_accuracy.numpy():.4f}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        print(f"Execution Time: {execution_time:.2f} seconds")

        # Plot loss and accuracy curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve (LR={learning_rate}, HL={n_hidden1},{n_hidden2})')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), test_accuracies, marker='o', color='r', label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Curve (LR={learning_rate}, HL={n_hidden1},{n_hidden2})')
        plt.legend()
        plt.show()
        
        # Confusion matrix
        y_pred = tf.argmax(forward_pass(x_test), axis=1).numpy()
        y_true = tf.argmax(y_test, axis=1).numpy()
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (LR={learning_rate}, HL={n_hidden1},{n_hidden2})')
        plt.show()
        
        print("============================================")
