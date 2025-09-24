# Step 1.1 - Load and Prepare Data
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Load dataset
data = load_wine()
X = data.data
y = data.target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print("Train set size:", X_train_tensor.shape)
print("Test set size:", X_test_tensor.shape)

# Step 1.2 - Visualize the Dataset with PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Wine Dataset Visualized with PCA')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.grid(True)
plt.show()

# Step 2.1 - Implement a Logistic Regression Model
import torch.nn as nn
import torch.optim as optim
import time

# Define logistic regression model
torch.manual_seed(42)
model = nn.Linear(X_train_tensor.shape[1], 3)  # 3 classes in wine dataset

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop with time measurement
start_time = time.time()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
lr_training_time = time.time() - start_time

print(f"Logistic Regression Training completed in {lr_training_time:.4f} seconds")

# Step 2.2 – Evaluate the Model
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = torch.argmax(predictions, dim=1)
    accuracy = (predicted_labels == y_test_tensor).float().mean()
    print(f"Logistic Regression Test Accuracy: {accuracy.item() * 100:.2f}%")

# Step 3.1 - Train a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Train decision tree with time measurement
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
start_time = time.time()
tree.fit(X_train, y_train)
dt_training_time = time.time() - start_time

print(f"Decision Tree Training completed in {dt_training_time:.4f} seconds")

# Evaluate on test data
y_pred = tree.predict(X_test)
tree_accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Test Accuracy: {tree_accuracy * 100:.2f}%")

# Step 4.1 - Define the Neural Network
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
torch.manual_seed(42)
nn_model = SimpleNN(input_dim=X_train_tensor.shape[1], hidden_dim=16, output_dim=3)

# Step 4.2 - Train the Neural Network
nn_optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
nn_criterion = nn.CrossEntropyLoss()

# Training with time measurement
start_time = time.time()
for epoch in range(100):
    nn_optimizer.zero_grad()
    output = nn_model(X_train_tensor)
    loss = nn_criterion(output, y_train_tensor)
    loss.backward()
    nn_optimizer.step()
nn_training_time = time.time() - start_time

print(f"Neural Network Training completed in {nn_training_time:.4f} seconds")

# Step 4.3 - Evaluate Neural Network Performance
with torch.no_grad():
    predictions = nn_model(X_test_tensor)
    predicted_labels = torch.argmax(predictions, dim=1)
    nn_accuracy = (predicted_labels == y_test_tensor).float().mean()
    print(f"Neural Network Test Accuracy: {nn_accuracy.item() * 100:.2f}%")

# Step 5.1: Create a Comparison Table
import pandas as pd
import numpy as np

# Create comprehensive results dictionary
results = {
    "Model": ["Logistic Regression", "Decision Tree", "Neural Network"],
    "Test Accuracy (%)": [f"{accuracy.item() * 100:.2f}", f"{tree_accuracy * 100:.2f}", f"{nn_accuracy.item() * 100:.2f}"],
    "Training Time (s)": [f"{lr_training_time:.4f}", f"{dt_training_time:.4f}", f"{nn_training_time:.4f}"],
    "Interpretability": ["High", "High", "Medium"],
    "Parameter Count": [
        sum(p.numel() for p in model.parameters()),
        "N/A (Non-parametric)",
        sum(p.numel() for p in nn_model.parameters())
    ],
    "Best For": ["Linear relationships", "Interpretable decisions", "Complex patterns"]
}

# Create DataFrame
results_df = pd.DataFrame(results)

# Display the results with nice formatting
print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# Additional insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"Dataset: Wine dataset ({X.shape[0]} samples, {X.shape[1]} features)")
print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
print(f"Number of classes: {len(np.unique(y))}")

# Find best model by accuracy
accuracies = [accuracy.item(), tree_accuracy, nn_accuracy.item()]
best_model_idx = np.argmax(accuracies)
best_model_name = results["Model"][best_model_idx]
best_accuracy = max(accuracies) * 100

print(f"\nBest performing model: {best_model_name} ({best_accuracy:.2f}% accuracy)")

# Additional comparison
print("\nModel Characteristics:")
print("- Logistic Regression: Simple, fast, highly interpretable")
print("- Decision Tree: Very interpretable, no feature scaling needed") 
print("- Neural Network: More complex, can capture non-linear patterns")
print("="*80)
