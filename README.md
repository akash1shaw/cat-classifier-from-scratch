# 🧠 Deep Neural Network from Scratch

This project contains a fully modular and vectorized implementation of a **Deep Neural Network (DNN)** from scratch using **NumPy**. It covers everything from forward/backward propagation to regularization, dropout, optimizers, and training on real datasets like **Cat vs Non-Cat** and **MNIST digits**.

---

## 📁 Folder Structure

| File                          | Description |
|------------------------------|-------------|
| `project_deep_NN.ipynb`      | Main notebook walking through model design, training, evaluation.  
| `MNISTdigit_calssification.ipynb` | Notebook for digit recognition using MNIST dataset.  
| `main.ipynb`                 | Training/inference notebook for experimentation.  
| `forward_propagation.py`     | Contains the code for forward pass (linear + activation).  
| `backward_propagation.py`    | Contains backpropagation logic for each layer.  
| `back_prop_with_dropout.py`  | Adds dropout regularization during backpropagation.  
| `regularized_back_prop.py`   | Adds L2 regularization in backpropagation.  
| `initialization.py`          | Layer weight and bias initialization strategies.  
| `mini_batch.py`              | Mini-batch generation for stochastic optimization.  
| `optimizers.py`              | Implements Gradient Descent, Momentum, and Adam optimizers.  
| `train_catvnoncat.h5`        | Training data for cat vs non-cat binary classification.  
| `test_catvnoncat.h5`         | Test data for cat vs non-cat.  
| `a.jpg`                      | A sample image for visual debugging or demo.  
| `README.md`                  | You’re reading it :)

---

## 🚀 Features

- Fully vectorized deep neural network (no frameworks)
- Supports multiple layers with flexible architecture
- Forward and backward propagation
- Dropout regularization
- L2 regularization
- Mini-batch gradient descent
- Optimizers: Gradient Descent, Momentum, Adam
- Modular codebase for easy reuse and experimentation
- Notebooks for both binary (cat vs non-cat) and multi-class (MNIST) classification

---

## 🧪 Datasets

- **Cat vs Non-Cat**:
  - Provided as `.h5` files: `train_catvnoncat.h5`, `test_catvnoncat.h5`
- **MNIST Digits**:
  - Downloaded in `MNISTdigit_calssification.ipynb` using Keras or `sklearn`.

---





