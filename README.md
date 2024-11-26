# Single-Layer Perceptron for Iris Dataset Classification

## Overview
This project implements a **single-layer perceptron** from scratch to classify the **Fisher Iris dataset**. The perceptron is trained using a binary classification task (Iris Setosa vs. Others) with custom preprocessing and metrics evaluation over training epochs.

---

## Features
- **Preprocessing Method:** Data is binarized using a specified threshold.
- **Classification Algorithm:** Single-layer perceptron with:
  - Threshold activation function.
  - Widrow-Hoff rule for weight updates.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- **Performance Graphs:** Metric trends over training epochs are plotted for analysis.

---

## Dataset
The dataset used is the **Fisher Iris dataset**, which contains 150 samples across three classes:
1. Iris Setosa
2. Iris Versicolor
3. Iris Virginica  

Features include:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

### Binary Classification Setup
- **Positive Class:** Iris Setosa (class 0).
- **Negative Class:** Iris Versicolor (class 1) and Iris Virginica (class 2).

---

## Project Workflow

### 1. Preprocessing
- Method: Binarization  
  Each feature is converted into binary values (0 or 1) based on a threshold:
  x_i = 1 if x_i >= threshold, else 0.
- Threshold used: **3.0** (adjustable).

---

### 2. Single-Layer Perceptron
The perceptron model consists of:
1. **Input:** 4 binary features from the preprocessed data.
2. **Weights:** Randomly initialized and updated during training.
3. **Activation Function:** Threshold function:
   y_pred = 1 if net >= 0.5, else 0.
4. **Learning Rule:** Widrow-Hoff rule:
   w_i = w_i + eta * delta * x_i

---

### 3. Training and Testing
- **Dataset Split:** The dataset is split into 2/3 for training and 1/3 for testing, ensuring all classes are represented in both sets.
- **Training:** The perceptron adjusts weights to minimize errors on the training set.
- **Testing:** Predictions on the test set are compared with ground truth to compute evaluation metrics.

---

### 4. Metrics Evaluation
Metrics used to evaluate performance:
1. **Confusion Matrix:**
   - True Positives (TP)
   - False Positives (FP)
   - False Negatives (FN)
   - True Negatives (TN)
2. **Accuracy:** Proportion of correct predictions.
3. **Precision:** Proportion of positive predictions that are correct.
4. **Recall:** Proportion of actual positives correctly identified.
5. **F1-score:** Harmonic mean of Precision and Recall.
6. **ROC-AUC:** Area under the Receiver Operating Characteristic curve.

---

## Requirements
- Python 3.7 or higher
- Libraries:
  - NumPy
  - Scikit-learn
  - Matplotlib

---

## How to Run
1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>


##to install dependencies:
pip install numpy scikit-learn matplotlib
