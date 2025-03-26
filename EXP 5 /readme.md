# Convolutional Neural Network (CNN) for Fashion MNIST Classification

## Objective
The objective of this project is to train and evaluate a convolutional neural network using Keras Library to classify MNIST fashion dataset.
Demonstrate the effect of filter size, regularization, batch size and optimization algorithm on model performance. 
---

## Dataset
We used the **Fashion MNIST dataset**, which contains 70,000 grayscale images of 10 different fashion categories. The dataset is split into:
- **60,000 training images**
- **10,000 test images**

The categories include: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

### Model Architecture:
The CNN model is structured as follows:
- **Conv2D Layer**: First convolutional layer with 32 filters, a kernel size of `(3,3)` or `(5,5)`, and ReLU activation.
- **MaxPooling2D Layer**: Pooling operation with a `(2,2)` window.
- **Conv2D Layer**: Second convolutional layer with 64 filters and ReLU activation.
- **MaxPooling2D Layer**: Another pooling layer.
- **Flatten Layer**: To flatten the output of the previous layer before feeding it into a dense layer.
- **Dense Layer**: Fully connected layer with 128 units and ReLU activation.
- **Dropout Layer**: Dropout with a rate of 0.5 to prevent overfitting.
- **Dense Layer**: Final output layer with 10 units and softmax activation to produce class probabilities.

The model uses **L2 regularization** to prevent overfitting and **dropout** to add further regularization.

---
## Code Description

### Libraries Used:
- **TensorFlow / Keras**: For building and training the CNN.
- **Matplotlib & Seaborn**: For visualizing the training progress, loss curves, and confusion matrix.
- **NumPy**: For numerical operations.
- **scikit-learn**: For evaluating the model using confusion matrix.
- **Google Colab's `files` module**: For downloading the generated plots.

### Code Flow:
1. **GPU Check**: The code first checks whether a GPU is available for training. If yes, it will use the GPU, otherwise, it will fall back to using the CPU.
2. **Data Loading & Preprocessing**: The Fashion MNIST dataset is loaded and normalized (scaling pixel values to [0, 1]).
3. **Model Architecture**:
    - Two convolutional layers with `ReLU` activation and L2 regularization.
    - Two max-pooling layers.
    - A dense layer with 128 units and ReLU activation.
    - A final softmax layer for classification.
4. **Hyperparameter Configuration**: The following hyperparameters are explored:
  - **Filter Sizes**: `(3, 3)` and `(5, 5)`
  - **Regularizations**: `0.0001` and `0.001`
  - **Batch Sizes**: `32` and `64`
  - **Optimizers**: `Adam` and `SGD`
  - **Dropout Rate**: `0.5`

5. **Training & Evaluation**: The model is trained for 10 epochs on each configuration, and its performance is evaluated using test accuracy.
6. **Results**:
   - The training and validation accuracy/loss curves are plotted.
   - A confusion matrix is generated for the best performing model (with the highest test accuracy).
7. **Downloadable Plots**: Accuracy curves, loss curves, and confusion matrices for the best model are saved and can be downloaded.

---

## Code Implementation:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
from google.colab import files

# Check GPU availability
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("Using GPU:", tf.config.list_physical_devices('GPU')[0])
else:
    print("No GPU found, training on CPU.")

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize the images to [0,1] range
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape data for CNN (Adding channel dimension)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Fashion MNIST Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Create directory to save plots
os.makedirs("plots", exist_ok=True)

# Experiment configuration parameters
filter_sizes = [(3, 3), (5, 5)]  # Different filter sizes for the convolutional layers
regularizations = [0.0001, 0.001]  # L2 regularization strengths
batch_sizes = [32, 64]  # Different batch sizes
optimizers = ['adam', 'sgd']  # Different optimizers
dropout_rate = 0.5  # Fixed dropout rate

# Function to create model with these configurations
def create_model(filter_size=(3, 3), regularization=0.0001, optimizer='adam', dropout_rate=0.5):
    model = keras.Sequential([
        layers.Conv2D(32, filter_size, activation='relu', kernel_regularizer=regularizers.l2(regularization), input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, filter_size, activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(regularization)),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model with the specified optimizer
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Experiment with different configurations
results = {}

for filter_size in filter_sizes:
    for regularization in regularizations:
        for batch_size in batch_sizes:
            for optimizer in optimizers:
                # Create a unique key for each configuration
                key = f"Filter={filter_size}, Reg={regularization}, Batch={batch_size}, Opt={optimizer}, Dropout={dropout_rate}"
                print(f"Training with {key}")
                    
                # Create and train the model with current configuration
                model = create_model(filter_size, regularization, optimizer, dropout_rate)
                history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)
                    
                # Evaluate the model
                test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                results[key] = (history, test_acc)  # Store history and test accuracy
                print(f"

plt.figure(figsize=(12, 8))
for key, (history, _) in results.items():
    plt.plot(history.history['val_accuracy'], label=key)
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Effect of Hyperparameters on Model Performance')
plt.legend(loc='upper right')
plt.show()
```
---

## Performance Evaluation

- The model's accuracy is tested with different hyperparameters to observe their effects.

- **Filter Size**: 
  - Affects feature extraction, with larger filters capturing more details but requiring more computations.

- **Regularization**: 
  - Helps in preventing overfitting but may reduce training accuracy.

- **Batch Size**: 
  - Smaller batch sizes may improve generalization but increase training time.

- **Optimizer Comparison**:
  - **Adam** generally provides better accuracy compared to **SGD**.
  - **SGD** may take longer to converge but can generalize well in some cases.

- **Loss Visualization**:
  - Training loss vs. validation loss is plotted over epochs to understand if the model overfits.
  - Helps determine the best epoch count for optimal performance.

- **Line Plot of Validation Accuracy**:
  - A line plot visualizes how validation accuracy changes with different configurations, helping identify the best-performing setup.
---
## My Comment:
- **Effect of Filter Size**:
    Larger filters tend to capture more complex features, but they make the training slower and increase the risk of overfitting.
- **Effect of Regularization**:
    Stronger regularization (larger L2 values) helps to prevent overfitting, especially when we are training with smaller filter sizes.
    Models with Reg=0.0001 perform better than those with Reg=0.001.
- **Effect of Batch Size**:
   Larger batch size (64) stabilizes training but may lead to slightly lower validation accuracy compared to batch size 32.
   Smaller batch size (32) allows for more weight updates per epoch, which can help generalization.
- **Effect of Optimizer**:
    Adam generally outperforms SGD due to its adaptive learning rate.

## Best-Performing Model

Based on the trends, the best-performing models have the following hyperparameters:

- **Filter Size**: 5
- **Regularization**: 0.0001
- **Batch Size**: 32
- **Optimizer**: Adam

These parameters yield the highest validation accuracy (~90%).

---
