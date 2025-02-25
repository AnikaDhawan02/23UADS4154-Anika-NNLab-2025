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
