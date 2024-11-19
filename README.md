# Decision Tree Implementation in Python

This repository contains a custom implementation of a **Decision Tree Classifier** from scratch in Python. The classifier supports basic functionality like fitting a dataset, predicting labels, and handling overfitting with parameters for minimum samples split and maximum depth.

---

## Features
- **Customizable hyperparameters**: `min_samples_split`, `max_depth`, and `n_features`.
- **Supports entropy-based splits** for information gain.
- **Random feature selection** for splits (useful for ensemble methods like Random Forest).
- **Pure Python implementation**: No external machine learning libraries used.

---

## Installation
Clone the repository and make sure you have Python installed with `numpy`.

```bash
git clone <your-repo-url>
cd <repo-name>
```


Usage
Here’s an example of how to use the decision tree:
```
import numpy as np
from decision_tree import DecisionTree

# Generate a toy dataset
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 0, 1, 1])

# Initialize the model
tree = DecisionTree(min_samples_split=2, max_depth=3)

# Train the model
tree.fit(X, y)

# Make predictions
predictions = tree.predict(X)
print("Predictions:", predictions)
```
Parameters
min_samples_split: Minimum number of samples required to split a node (default: 2).
max_depth: Maximum depth of the tree (default: 100).
n_features: Number of features to consider for the best split (default: all features).

