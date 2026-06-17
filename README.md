# MicroML

A lightweight Machine Learning library built entirely from scratch using NumPy.

MicroML reimplements core machine learning algorithms without relying on machine learning frameworks such as scikit-learn, TensorFlow, or PyTorch. The project focuses on understanding the mathematics, optimization techniques, and implementation details behind modern machine learning systems.

---

## Features

### Metrics

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Accuracy
* R² Score

### Linear Models

* Linear Regression
* Logistic Regression

### Optimizers

* Stochastic Gradient Descent (SGD)
* Momentum
* Adam

### Classification

* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes

### Dimensionality Reduction

* Principal Component Analysis (PCA)

### Clustering

* K-Means Clustering

---

## Project Structure

```text
MicroML/
│
├── microml/
│   ├── metrics/
│   ├── linear_model/
│   ├── neighbors/
│   ├── bayes/
│   ├── decomposition/
│   ├── clustering/
│   ├── optimizers/
│   └── utils/
│
├── datasets/
├── notebooks/
├── tests/
├── benchmarks/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Implemented Algorithms

| Category                 | Algorithm            |
| ------------------------ | -------------------- |
| Regression               | Linear Regression    |
| Classification           | Logistic Regression  |
| Classification           | K-Nearest Neighbors  |
| Classification           | Gaussian Naive Bayes |
| Dimensionality Reduction | PCA                  |
| Clustering               | K-Means              |
| Optimization             | SGD                  |
| Optimization             | Momentum             |
| Optimization             | Adam                 |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/MicroML.git

cd MicroML

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

---

## Example Usage

```python
import numpy as np

from microml.linear_model.linear_regression import LinearRegression

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([
    3,
    5,
    7,
    9,
    11
])

model = LinearRegression(
    learning_rate=0.01,
    epochs=1000
)

model.fit(X, y)

predictions = model.predict(X)

print(predictions)
```

---

## Learning Objectives

This project was created to understand:

* Gradient Descent
* Optimization Algorithms
* Probability and Statistics
* Linear Algebra
* Classification Techniques
* Clustering Algorithms
* Dimensionality Reduction
* Object-Oriented Design
* Numerical Computing with NumPy

---

## Current Status

### Version 1.0 Completed

* [x] Metrics Module
* [x] Linear Regression
* [x] Logistic Regression
* [x] KNN
* [x] Gaussian Naive Bayes
* [x] PCA
* [x] K-Means
* [x] SGD
* [x] Momentum
* [x] Adam

### Upcoming

* [ ] Interactive Dashboard
* [ ] Visualization Tools
* [ ] Benchmarking Against Scikit-Learn
* [ ] Performance Analysis
* [ ] Documentation Improvements

---

## Why This Project?

Most machine learning libraries hide the implementation details of algorithms.

MicroML exposes the underlying mathematics and optimization process by implementing every algorithm from scratch using only NumPy. The project serves as both a learning resource and a foundation for experimenting with machine learning fundamentals.

---

## License

MIT License

---






Machine Learning • Deep Learning • AI Engineering
