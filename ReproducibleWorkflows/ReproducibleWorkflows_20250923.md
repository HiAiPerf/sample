Demonstrate how to create reproducible machine learning workflows using TensorFlow/Keras.

## Step 1: Import Libraries
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import joblib
```
- **numpy**: For numerical operations
- **tensorflow/keras**: For building neural networks
- **sklearn**: For data generation, splitting, and preprocessing
- **random**: For setting random seeds
- **joblib**: For saving/loading Python objects

## Step 2: Set Random Seeds for Reproducibility
```python
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```
**Purpose**: Ensures that random operations produce the same results every time the code runs. This is crucial for reproducible machine learning.

## Step 3: Generate Synthetic Data
```python
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15,
    n_redundant=5, 
    n_classes=2, 
    random_state=42
)
```
**Parameters**:
- `n_samples=1000`: Creates 1000 data points
- `n_features=20`: Each sample has 20 features
- `n_informative=15`: 15 features are actually useful for classification
- `n_redundant=5`: 5 features are redundant combinations of informative features
- `n_classes=2`: Binary classification problem
- `random_state=42`: Ensures same data is generated every time

**Output**: `X` contains features, `y` contains binary labels (0 or 1)

## Step 4: Split and Scale Data
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
**Process**:
1. **Split**: 80% training data, 20% test data
2. **Scale**: Standardize features to have mean=0 and variance=1
   - `fit_transform()`: Learns scaling parameters from training data and applies them
   - `transform()`: Applies same scaling to test data (without re-learning parameters)

## Step 5: Save Data
```python
joblib.dump((X_train, y_train), 'train_data.pkl')
joblib.dump((X_test, y_test), 'test_data.pkl')
```
**Purpose**: Saves the processed data for future use, ensuring exact same data can be reloaded.

## Step 6: Build Neural Network Model
```python
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
```
**Architecture**:
- **Input layer**: 20 neurons (matches number of features)
- **Hidden layer 1**: 32 neurons with ReLU activation
- **Hidden layer 2**: 16 neurons with ReLU activation
- **Output layer**: 1 neuron with sigmoid activation (for binary classification)

## Step 7: Compile Model
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```
**Configuration**:
- **Optimizer**: Adam (adaptive learning rate optimizer)
- **Loss function**: Binary crossentropy (for binary classification)
- **Metric**: Accuracy to monitor training progress

## Step 8: Train Model
```python
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
```
**Training parameters**:
- `epochs=20`: Process entire dataset 20 times
- `batch_size=32`: Process 32 samples at a time before updating weights
- `validation_split=0.1`: Use 10% of training data for validation during training

## Step 9: Evaluate Model
```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")
```
**Purpose**: Tests the model on unseen test data to measure generalization performance.

## Step 10: Save Model and Scaler
```python
model.save('my_model.keras')
```
**Purpose**: Saves the entire trained model architecture, weights, and training configuration.

## Step 11: Reload and Verify Reproducibility
```python
from tensorflow.keras.models import load_model
import joblib

modelReloaded = load_model('my_model.keras')
X_train_reloaded, y_train_reloaded = joblib.load('train_data.pkl')
X_test_reloaded, y_test_reloaded = joblib.load('test_data.pkl')

loss, accuracy = modelReloaded.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")
```
**Verification**: Demonstrates that the saved model produces identical results when reloaded.

## Key Reproducibility Features:

1. **Random seeds**: Control randomness in data generation, splitting, and neural network initialization
2. **Data persistence**: Save/Load exact same datasets
3. **Model persistence**: Save/Load entire trained model
4. **Deterministic operations**: TensorFlow operations become reproducible with fixed seeds

This workflow ensures that anyone running this code will get exactly the same results, which is essential for scientific experiments and production systems.