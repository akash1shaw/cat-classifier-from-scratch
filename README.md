# cat-classifier-from-scratch
Binary image classification using a deep neural network from scratch (no ML libraries)
# 🐱 Cat Classifier from Scratch (No ML Libraries)

This project implements a deep neural network **from scratch using NumPy** to classify images as **cat vs non-cat**.


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




