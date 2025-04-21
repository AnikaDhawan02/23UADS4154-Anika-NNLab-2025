## Medical CT Scan Classification using VGG16
---
## Objective 

 The main objective of this program is to retrain a pretrained imagenet model to classify a medical image dataset. 
 
## Model Descrpition
 
The dataset consists of CT scan images stored in two separate folders (covid and non_covid).
Base Model: VGG16 (pretrained on ImageNet)

Modification:

+ The final fully connected layer was replaced to have 2 output neurons (COVID vs Non-COVID classification).

Training Strategy:

+ Transfer learning: Use pretrained VGG16 weights.

+ Fine-tuning the model with a small learning rate.

Loss Function: CrossEntropyLoss
Optimizer: Adam
Frameworks Used: PyTorch, torchvision

## Python implementation 
```
# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Preparation
data_dir = r"C:\Users\Dell\OneDrive\Desktop\NN LAB EXP\medical data"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load Pretrained VGG16 Model
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 2)  # Replace final FC layer for 2 classes
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {accuracy:.2f}%")

# Save the Trained Model
os.makedirs('saved_models', exist_ok=True)
torch.save(model.state_dict(), 'saved_models/vgg16_covid_classifier.pth')

```
## Description of code

* Imports: PyTorch and torchvision for deep learning and image processing.
* Device Setup: Use GPU if available.
* Data Preparation: Load CT scan images (COVID/Non-COVID), resize, normalize, and split into train/validation.
* Model: Load pretrained VGG16, replace final layer with 2 output classes.
* Loss/Optimizer: Use CrossEntropyLoss and Adam optimizer.
* Training: Train for 10 epochs, track loss and validation accuracy.
* Saving: Save the trained model to disk.


## Output
During training, you will see outputs like:
```
Epoch [1/10], Loss: 0.5932, Validation Accuracy: 78.50%
Epoch [2/10], Loss: 0.4234, Validation Accuracy: 85.00%
Epoch [3/10], Loss: 0.3121, Validation Accuracy: 89.50%
...
Epoch [10/10], Loss: 0.0915, Validation Accuracy: 97.00%
Model saved successfully!
```

* Low loss and high accuracy show strong learning.
* Transfer learning helped due to small dataset size.
  
## My Comment 

* it uses the transfer learning on VGG16 to improve generalization and
* accelerate convergence, addressing the limited size of the medical CT dataset.
* add data augmentation and dynamic learning rates for even better performance

