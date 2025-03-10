# **README.md**

## **Objective**

WAP to evaluate the performance of implemented three-layer neural network with variations in activation functions, size of hidden layer, learning rate, batch size and number of epochs. 
---

## **Description of the Model**

The model is a **feedforward neural network** with the following architecture:
- **Layer 1:** Varies based on hidden layer configurations (e.g., 160, 100; 100, 100; etc.) with **ReLU** activation.
- **Layer 2:** Varies based on hidden layer configurations (e.g., 100, 100; 160, 100; etc.) with **ReLU** activation.
- **Output Layer:** 10 neurons (representing digits 0-9) with **softmax** activation.

### **Hyperparameters:**
- **Learning Rate Variations:** 0.1, 0.01, 1
- **Hidden Layer Configurations:** (160, 100), (100, 100), (100, 160), (60, 60), (100, 60)
- **Activation Functions:** ReLU for hidden layers, Softmax for the output layer
- **Batch Size:** Fixed at **10**
- **Epochs:** Fixed at **20**

The performance is evaluated based on **accuracy**, **loss curves**, and **confusion matrices** for each combination of hyperparameters.
---

## **Training Process**

- The model is trained with different **combinations** of hidden layer sizes, activation functions, and learning rates.
- **Adam Optimizer** is used for training with a learning rate of 0.01 (by default).
- **Loss Function:** Softmax cross-entropy loss is used for classification.
- The dataset is preprocessed by:
  - Normalizing pixel values (0 to 1)
  - Flattening the images into 784-element vectors
  - One-hot encoding the labels

---

## **Code Implementation**

```python
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

...
```

### **Key Functionality:**
1. **Model Architecture:** The model consists of two hidden layers with varying numbers of neurons, followed by a softmax output layer.
2. **Hyperparameter Combinations:** The model trains and evaluates over multiple combinations of hidden layer sizes, activation functions, and learning rates.
3. **Confusion Matrix:** After training, the model generates confusion matrices to assess classification performance.
4. **Loss & Accuracy Curves:** The code also plots training loss and accuracy curves for each configuration.
5. **Saving Results:** All results, including accuracy, confusion matrix, and loss/accuracy curves, are saved as images and in an Excel file for further analysis.

---

## **Performance Evaluation**

### **Metrics:**
1. **Test Accuracy:** The model's final accuracy is calculated on the test dataset after training.
2. **Confusion Matrix:** A confusion matrix is generated for each configuration to visualize classification errors across all digit classes.
3. **Loss Curve:** This curve plots how the training loss decreases over epochs.
4. **Accuracy Curve:** Both training and validation accuracy curves are generated to monitor the model's performance during training.

---

## **Results**

The results of training with different hidden layer configurations and learning rates are saved in:
- **Excel File:** `training_results.xlsx` which contains:
  - Hidden Layer Configurations
  - Learning Rate
  - Execution Time (seconds)
  - Test Accuracy
  - Confusion Matrix Image
  - Loss/Accuracy Curves Image

---

## **Configurations Tested**

1. **Hidden Layer Sizes:**
   - (160, 100)
   - (100, 100)
   - (100, 160)
   - (60, 60)
   - (100, 60)

2. **Learning Rates:**
   - 0.01
   - 0.1
   - 1

3. **Activation Functions:**
   - ReLU for hidden layers
   - Softmax for the output layer

4. **Batch Size:** Fixed at 10
5. **Epochs:** Fixed at 20

---

## **Comments**

- **TensorFlow 1.x Deprecation:** The code uses TensorFlow 1.x, which is now deprecated. Consider upgrading to TensorFlow 2.x for better performance, maintainability, and long-term support.
  
- **Weight Initialization:** The current code uses `tf.random_normal()` for weight initialization, which may not be ideal. Consider using more modern initializers such as `tf.keras.initializers.HeNormal()` or `tf.keras.initializers.GlorotUniform()` for better convergence and performance.

- **Data Loading Efficiency:** The code uses `tf.compat.v1.data.make_one_shot_iterator` for training and testing. For better performance and efficiency, it is recommended to use the `tf.data` pipeline with `repeat()` and `prefetch()` methods to load data asynchronously and in batches.

- **Validation Set:** The current implementation evaluates accuracy on the training set itself, which can lead to overfitting. A separate **validation set** should be used to track the model's generalization ability during training.

- **Learning Rate Optimization:** A fixed learning rate (0.01) is used throughout training. In practice, it is often beneficial to use an adaptive learning rate schedule, such as `ReduceLROnPlateau` or a learning rate decay, to help the model converge better and faster.

- **Training Stability:** The model trains using a **batch size of 10**, which is relatively small. Smaller batch sizes might cause more noise in the gradient updates, resulting in unstable training. Experimenting with larger batch sizes or using **gradient accumulation** for small batch sizes could improve stability and convergence.

- **Model Overfitting:** Given that the dataset is MNIST (a relatively simple dataset), the model might not generalize well to more complex datasets. More advanced techniques like **regularization** (dropout, L2 regularization) or data augmentation might be required for larger or more complex datasets.

---

## **Future Enhancements**

- **Upgrade to TensorFlow 2.x:** The code should be refactored to utilize TensorFlow 2.x for better support and improved performance.
  
- **Hyperparameter Tuning:** For further performance improvement, consider using **Grid Search** or **Random Search** techniques to systematically tune hyperparameters like learning rate, hidden layer sizes, and activation functions.

- **Optimization Algorithms:** Explore alternative optimization algorithms such as **SGD** with momentum, **RMSProp**, or **Adagrad**, which might improve training performance depending on the problem.

- **Data Augmentation:** Enhance the model's ability to generalize on unseen data by applying **data augmentation** techniques (e.g., rotations, scaling, random translations) to the training data.

- **Advanced Models:** Experiment with deeper architectures (more hidden layers) and more advanced neural network types (e.g., Convolutional Neural Networks) for better performance on more complex datasets.

---

## **Conclusion**

This project evaluates a simple three-layer neural network on the MNIST dataset by experimenting with different hyperparameters such as hidden layer sizes, learning rates, and activation functions. The model's performance is evaluated in terms of test accuracy, confusion matrix, and training curves. Future improvements could include updating the model to TensorFlow 2.x, optimizing hyperparameters, and experimenting with more advanced architectures and techniques.

---
