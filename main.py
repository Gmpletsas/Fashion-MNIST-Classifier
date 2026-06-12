# Import required libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np

# Load the Fashion MNIST dataset from Keras
# It contains 28x28 grayscale images of clothing items
fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Check dataset shapes and data types
print(X_train_full.shape, X_train_full.dtype)
print(y_train_full.shape, y_train_full.dtype)
print(X_test.shape, X_test.dtype)
print(y_test.shape, y_test.dtype)

# Class labels for the dataset (0–9 mapped to clothing types)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Print class mapping
for label, class_name in enumerate(class_names):
    print(f"Label {label}: {class_name}")

# Visualize first 9 images from the dataset
for i in range(9):
    pyplot.subplot(3, 3, i + 1)
    pyplot.imshow(X_train_full[i], cmap=pyplot.get_cmap('gray'))

pyplot.show()

# Split dataset into training and validation sets
# Also normalize pixel values from 0–255 to 0–1
X_valid, X_train = X_train_full[:5000] / 255, X_train_full[5000:] / 255
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Normalize test set as well
X_test = X_test / 255

# Build a Sequential neural network model
model = keras.models.Sequential()

# Flatten 28x28 images into a 1D vector (784 values)
model.add(keras.layers.Flatten(input_shape=[28, 28]))

# First hidden layer with 300 neurons and ReLU activation
model.add(keras.layers.Dense(300, activation="relu"))

# Second hidden layer with 100 neurons and ReLU activation
model.add(keras.layers.Dense(100, activation="relu"))

# Output layer with 10 neurons (one per class) using softmax
model.add(keras.layers.Dense(10, activation="softmax"))

# Display model architecture
model.summary()

# Access model layers
my_layers = model.layers

# Extract first hidden layer
first_hidden_layer = my_layers[1]

# Get weights and biases of first hidden layer
weights, biases = first_hidden_layer.get_weights()

# Compile the model (define loss function, optimizer, and metrics)
model.compile(
    loss="sparse_categorical_crossentropy",  # for integer labels
    optimizer="sgd",                          # stochastic gradient descent
    metrics=["accuracy"]                      # track accuracy
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_data=(X_valid, y_valid)
)

# Plot training history (loss + accuracy over epochs)
pd.DataFrame(history.history).plot(figsize=(8, 5))
pyplot.grid(True)
pyplot.gca().set_ylim(0, 1)
pyplot.title('Training History')

# Evaluate model performance on test data
model.evaluate(X_test, y_test)

# Take first 3 test images for prediction
X_new = X_test[:3]

# Predict class probabilities
y_proba = model.predict(X_new)
print(y_proba.round(2))

# Get index of highest probability for each prediction
class_names_idx = np.argmax(y_proba, axis=1)

# Convert numeric predictions into class names
result = [class_names[i] for i in class_names_idx]

# Print final predictions
print(result)
plt.show()
