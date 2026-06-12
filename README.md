# 🧠 Fashion MNIST Neural Network Classifier

This project builds a deep learning model using TensorFlow and Keras to classify clothing images from the Fashion MNIST dataset.

---

## 📊 Dataset

The model uses the **Fashion MNIST dataset**, which contains:
- 60,000 training images
- 10,000 test images
- 10 clothing categories (e.g., T-shirt, Trouser, Sneaker, Bag)

Each image is grayscale and 28×28 pixels.

---

## ⚙️ Model Architecture

The neural network is built using a simple feedforward (Sequential) model:

- Flatten layer (28×28 → 784 inputs)
- Dense layer (300 neurons, ReLU)
- Dense layer (100 neurons, ReLU)
- Output layer (10 neurons, Softmax)

---

## 🧪 Training Process

- Loss function: `sparse_categorical_crossentropy`
- Optimizer: `SGD`
- Metric: `accuracy`
- Epochs: 30
- Validation split: 5,000 images

---

## 📈 Features

- Loads and visualizes dataset samples
- Normalizes pixel values (0–255 → 0–1)
- Splits training and validation data
- Trains a neural network classifier
- Plots training vs validation performance
- Evaluates model on test data
- Makes predictions on new samples

---

## 🔍 Predictions

The model outputs probabilities for each clothing class and returns the final predicted label such as:

- Ankle boot
- Sneaker
- T-shirt/top
- Bag

---

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
