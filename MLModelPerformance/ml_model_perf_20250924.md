Explain the complete code line by line with special emphasis on PyTorch concepts.

## Step 1.1 - Load and Prepare Data

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
```
**Imports**: 
- `load_wine`: Loads the Wine dataset (chemical analysis of wines from 3 different cultivars)
- `train_test_split`: Splits data into training and testing sets
- `StandardScaler`: Standardizes features by removing mean and scaling to unit variance
- `torch`: PyTorch library for deep learning

```python
# Load dataset
data = load_wine()
X = data.data  # Feature matrix: 178 samples × 13 chemical features
y = data.target  # Target labels: 3 classes (wine types 0, 1, 2)
```
**Wine Dataset**:
- 13 features: alcohol, malic acid, ash, alkalinity, magnesium, phenols, flavonoids, etc.
- 3 classes: three different cultivars (types) of wine
- 178 total samples

```python
# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
**StandardScaler**:
- `fit_transform()`: Calculates mean/std of each feature, then transforms data
- Formula: `(x - mean) / std`
- **Why**: Neural networks converge faster when features are on similar scales

```python
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```
**Train-Test Split**:
- 80% training (142 samples), 20% testing (36 samples)
- `random_state=42`: Ensures reproducible splits

```python
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
```
**PyTorch Tensors**:
- `torch.tensor()`: Converts NumPy arrays to PyTorch tensors
- `dtype=torch.float32`: Features need float type for calculations
- `dtype=torch.long`: Labels need integer type for classification
- **Tensors**: PyTorch's equivalent of NumPy arrays, but with GPU support and automatic differentiation

## Step 1.2 - Visualize with PCA

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```
**PCA (Principal Component Analysis)**:
- Reduces 13-dimensional data to 2D while preserving most variance
- `n_components=2`: Keep only the 2 most important dimensions

The visualization shows how well the three wine classes separate in 2D space.

## Step 2.1 - Logistic Regression Model

```python
import torch.nn as nn
import torch.optim as optim
import time

# Define logistic regression model
torch.manual_seed(42)
model = nn.Linear(X_train_tensor.shape[1], 3)  # 3 classes in wine dataset
```
**PyTorch Model**:
- `torch.manual_seed(42)`: Sets random seed for reproducibility
- `nn.Linear(input_size, output_size)`: Creates a linear layer
  - Input: 13 features → Output: 3 classes
  - **Weights**: Matrix of shape (13, 3)
  - **Bias**: Vector of shape (3,)
- This is equivalent to logistic regression for multi-class classification

```python
# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```
**Loss Function & Optimizer**:
- `nn.CrossEntropyLoss()`: Standard loss for classification
  - Combines softmax activation + cross-entropy loss
  - Measures how well predicted probabilities match true labels
- `optim.SGD()`: Stochastic Gradient Descent optimizer
  - `model.parameters()`: All weights and biases of the model
  - `lr=0.01`: Learning rate (step size for weight updates)

```python
# Training loop
start_time = time.time()
for epoch in range(100):
    optimizer.zero_grad()      # Clear previous gradients
    outputs = model(X_train_tensor)  # Forward pass
    loss = criterion(outputs, y_train_tensor)  # Calculate loss
    loss.backward()            # Backward pass (compute gradients)
    optimizer.step()           # Update weights
lr_training_time = time.time() - start_time
```
**Training Loop Explained**:

1. **`optimizer.zero_grad()`**: 
   - PyTorch accumulates gradients by default
   - Must zero them before each backward pass to prevent accumulation

2. **Forward Pass**:
   ```python
   outputs = model(X_train_tensor)
   ```
   - Input: (142, 13) tensor → Output: (142, 3) tensor
   - Each row contains "scores" for the 3 classes (not probabilities yet)

3. **Loss Calculation**:
   ```python
   loss = criterion(outputs, y_train_tensor)
   ```
   - `outputs`: Predicted scores (142 samples × 3 classes)
   - `y_train_tensor`: True labels (142 integers: 0, 1, or 2)

4. **Backward Pass**:
   ```python
   loss.backward()
   ```
   - **Automatic Differentiation**: PyTorch automatically computes gradients of loss with respect to all parameters
   - Creates `.grad` attribute for each parameter tensor

5. **Weight Update**:
   ```python
   optimizer.step()
   ```
   - Updates weights using gradient descent: `weight = weight - lr * gradient`

## Step 2.2 - Evaluate Logistic Regression

```python
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = torch.argmax(predictions, dim=1)
    accuracy = (predicted_labels == y_test_tensor).float().mean()
```
**Evaluation Mode**:
- `with torch.no_grad()`: Disables gradient computation (faster, uses less memory)
- `torch.argmax(predictions, dim=1)`: Gets class with highest score for each sample
- `dim=1`: Take maximum across columns (across the 3 classes for each sample)

## Step 3.1 - Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
tree_accuracy = accuracy_score(y_test, y_pred)
```
**Decision Tree**:
- Non-parametric model that makes decisions based on feature thresholds
- `max_depth=5`: Prevents overfitting by limiting tree depth
- No need for feature scaling (tree splits are based on thresholds)

## Step 4.1 - Neural Network Definition

```python
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input → Hidden
        self.fc2 = nn.Linear(hidden_dim, output_dim) # Hidden → Output

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)          # Output layer (no activation)
        return x
```
**Neural Network Class**:

1. **`nn.Module`**: Base class for all PyTorch neural networks
2. **`__init__`**: Defines layers
   - `fc1`: First fully connected layer (13 → 16 neurons)
   - `fc2`: Second fully connected layer (16 → 3 neurons)
3. **`forward`**: Defines how data flows through the network
   - `F.relu()`: Activation function (Rectified Linear Unit)
     - Formula: `max(0, x)`
     - Introduces non-linearity so network can learn complex patterns

```python
# Initialize model
torch.manual_seed(42)
nn_model = SimpleNN(input_dim=X_train_tensor.shape[1], hidden_dim=16, output_dim=3)
```

## Step 4.2 - Train Neural Network

```python
nn_optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
nn_criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    nn_optimizer.zero_grad()
    output = nn_model(X_train_tensor)  # Forward pass through NN
    loss = nn_criterion(output, y_train_tensor)
    loss.backward()
    nn_optimizer.step()
```
**Adam Optimizer**:
- More sophisticated than SGD
- Adapts learning rates for each parameter
- Often converges faster

## Step 5.1 - Comparison Table

```python
import pandas as pd
import numpy as np

results = {
    "Model": ["Logistic Regression", "Decision Tree", "Neural Network"],
    "Test Accuracy (%)": [f"{accuracy.item() * 100:.2f}", 
                         f"{tree_accuracy * 100:.2f}", 
                         f"{nn_accuracy.item() * 100:.2f}"],
    "Training Time (s)": [f"{lr_training_time:.4f}", 
                         f"{dt_training_time:.4f}", 
                         f"{nn_training_time:.4f}"],
    "Parameter Count": [
        sum(p.numel() for p in model.parameters()),  # Count all parameters
        "N/A (Non-parametric)",
        sum(p.numel() for p in nn_model.parameters())
    ]
}
```

**Key PyTorch Concepts Used**:

1. **`tensor.item()`**: Extract scalar value from single-element tensor
2. **`sum(p.numel() for p in model.parameters())`**: Count total parameters
   - `p.numel()`: Number of elements in parameter tensor
   - Logistic Regression: 13×3 weights + 3 biases = 42 parameters
   - Neural Network: (13×16 + 16) + (16×3 + 3) = 239 parameters

## Key PyTorch Takeaways:

1. **Tensors**: Fundamental data structure (like NumPy arrays + GPU support)
2. **Autograd**: Automatic differentiation for gradient computation
3. **nn.Module**: Base class for all neural networks
4. **Optimizers**: Update parameters based on gradients
5. **Training Loop**: Forward pass → Loss calculation → Backward pass → Update

This code demonstrates a complete machine learning workflow comparing simple vs complex models on a real dataset!
