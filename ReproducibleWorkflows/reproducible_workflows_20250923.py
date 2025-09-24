import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import joblib

# Creating Reproducible Datasets

# 1. Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# 2. Generate synthetic data
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15,
    n_redundant=5, 
    n_classes=2, 
    random_state=42
)

# Training Reproducible Model

# 3. Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# save the train and test data 
joblib.dump((X_train, y_train), 'train_data.pkl')
joblib.dump((X_test, y_test), 'test_data.pkl')

## Model Initalization and Training
# 4. Build a neural network
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# 6. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")


# Saving Models

# 7. Save the model and scaler
model.save('my_model.keras')

# Reproducing Models

#8, Reloading model later
from tensorflow.keras.models import load_model
import joblib
modelReloaded = load_model('my_model.keras')
X_train_reloaded, y_train_reloaded = joblib.load('train_data.pkl')
X_test_reloaded, y_test_reloaded = joblib.load('test_data.pkl')

loss, accuracy = modelReloaded.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")


loss, accuracy = modelReloaded.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")
