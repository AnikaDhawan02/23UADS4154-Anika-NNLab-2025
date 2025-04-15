# Recurrent Neural Network (RNN) 

## Objective

The objective of this project is to train and evaluate a Recurrent Neural Network using PyTorch Library to predict the next value in a sample time series dataset. 

## Model Overview

The architecture used in this project consists of:

  + Input Layer: Receives sequences of past sine wave values.
  + RNN Layer: A basic RNN with 64 hidden units to capture temporal dependencies.
  + Output Layer: A fully connected layer to predict the next time step value.
  + Loss Function: Mean Squared Error (MSE)
  + Optimizer: Adam

## Implementation Details
1. Data Generation and Preprocessing

- A sine wave is generated using NumPy to simulate time series data.
- The data is scaled using MinMaxScaler to the range [-1, 1].
- A custom PyTorch Dataset class is used to create sequences for training.

2. Model Architecture
```
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
```
3. Training Process
 - Model is trained for 50 epochs on sequence windows of length 20.
 - Loss is calculated using MSE and updated with Adam optimizer.
 - Epoch-wise training loss is printed to monitor learning progress.

4. Evaluation Metrics
* Mean Absolute Error (MAE) is used to assess model precision.
* A custom-defined accuracy metric calculates the percentage of predictions within 5% of true values.
  
## Python Implementation

```

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Create synthetic time series (sine wave)
time_steps = 1000
t = np.linspace(0, 100, time_steps)
data = np.sin(t)

# Normalize the data (between 0 and 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# Convert to PyTorch tensors
data_scaled = torch.FloatTensor(data_scaled).view(-1)

# Split data into training and test sets
train_size = int(len(data_scaled) * 0.8)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

# Create Dataset class for Time Series Data
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_length=10):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return x, y

# Create datasets and DataLoader
seq_length = 20  # Sequence length
train_dataset = TimeSeriesDataset(train_data, seq_length)
test_dataset = TimeSeriesDataset(test_data, seq_length)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)  # Output from RNN (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # We only care about the last time step
        return out

# Initialize the model, loss function, and optimizer
model = RNNModel(input_size=1, hidden_size=64, output_size=1, num_layers=1)
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.unsqueeze(-1)
        y_batch = y_batch.unsqueeze(-1)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

# Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
test_preds = []
test_true = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.unsqueeze(-1)
        y_batch = y_batch.unsqueeze(-1)

        y_pred = model(x_batch)
        test_preds.append(y_pred.numpy())
        test_true.append(y_batch.numpy())

# Convert lists to numpy arrays
test_preds = np.concatenate(test_preds, axis=0)
test_true = np.concatenate(test_true, axis=0)

# Inverse transform the data to get the original scale
test_preds = scaler.inverse_transform(test_preds)
test_true = scaler.inverse_transform(test_true)

# Flatten the arrays if they have an extra dimension
test_preds = test_preds.flatten()
test_true = test_true.flatten()

# Calculate MAE (Mean Absolute Error)
mae = mean_absolute_error(test_true, test_preds)
print(f"Mean Absolute Error (MAE) on Test Set: {mae:.4f}")

# Calculate accuracy as percentage of predictions within 5% of the true value
def calculate_accuracy(y_true, y_pred, tolerance=0.05):
    # Calculate absolute percentage error
    abs_percentage_error = np.abs((y_true - y_pred) / y_true)

    # Count how many predictions are within the tolerance range
    accuracy = np.sum(abs_percentage_error <= tolerance) / len(y_true) * 100
    return accuracy

accuracy = calculate_accuracy(test_true, test_preds, tolerance=0.05)
print(f"Accuracy (within 5% tolerance) on Test Set: {accuracy:.2f}%")

# Plot the results with different line styles
plt.figure(figsize=(10, 6))
plt.plot(test_true, label='True Values', color='blue', linestyle='-')  # Solid line for true values
plt.plot(test_preds, label='Predicted Values', color='red', linestyle='--')  # Dashed line for predicted values
plt.legend()
plt.title("True vs Predicted Values on Test Set")
plt.show()

# Alternatively, plot the difference between true and predicted values
plt.figure(figsize=(10, 6))
plt.plot(test_true - test_preds, label='Difference (True - Predicted)', color='green')
plt.legend()
plt.title("Difference between True and Predicted Values on Test Set")
plt.show()

```

## Results and Visualization
The model closely approximates sine wave behavior on the test set.

* Visualization includes:

      Line Plot of predicted vs true values
      Error Plot showing the difference between actual and predicted values
      The use of a custom accuracy metric provides a better sense of prediction reliability within tolerance.


## My Comments
- RNN is effective for short-term pattern recognition in clean, continuous data.
- For capturing long-range dependencies or dealing with noisy datasets, consider replacing RNN with LSTM or GRU models.
- Model assumes clean sine wave data — might not perform as well on noisy or irregular data.



