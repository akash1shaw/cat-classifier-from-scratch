# 🐱 Cat Classifier from Scratch (No ML Libraries)

This project implements a deep neural network **from scratch using NumPy** to classify images as **cat vs non-cat**.

![Model Architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Artificial_neural_network.svg/600px-Artificial_neural_network.svg.png)

---

## 📌 Features

- Forward and backward propagation manually implemented
- Supports ReLU, Sigmoid, and Tanh activations
- Trains on `catvnoncat.h5` dataset from Andrew Ng's DL course
- Visualizes misclassified test examples
- Can predict on custom user-supplied image

---

## 🧠 Model Architecture

- Input layer: 12288 (64×64×3)
- Hidden layers: Configurable
- Output: Sigmoid activation for binary classification

---

## 🚀 Training Results

- **Train Accuracy**: 98%
- **Test Accuracy**: 70–75%
- Optimized using full-batch gradient descent
- Cost plotted to show convergence

---

## 🖼️ Example Output

