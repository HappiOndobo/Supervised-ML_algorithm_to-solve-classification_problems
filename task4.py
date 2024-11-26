import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
data, target = iris.data, iris.target

# Preprocessing Method 9: Binarization
threshold = 3.0  # Example threshold for binarization
binarizer = Binarizer(threshold=threshold)
processed_data = binarizer.fit_transform(data)

# Split data into training and testing sets (2/3 training, 1/3 testing)
train_data, test_data, train_target, test_target = train_test_split(
    processed_data, target, test_size=0.33, random_state=42, stratify=target
)

# Convert targets to binary classification (Setosa vs others)
train_target_binary = (train_target == 0).astype(int)
test_target_binary = (test_target == 0).astype(int)

# Define Single-Layer Perceptron
class SingleLayerPerceptron:
    def __init__(self, num_features, learning_rate=0.1):
        self.weights = np.random.uniform(-1, 1, num_features)
        self.bias = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.metrics_per_epoch = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    def activation(self, net):
        return 1 if net >= 0.5 else 0  # Threshold activation

    def predict(self, inputs):
        net = np.dot(inputs, self.weights) + self.bias
        return self.activation(net)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            predictions = []
            total_error = 0
            for i in range(len(X)):
                y_pred = self.predict(X[i])
                predictions.append(y_pred)
                error = y[i] - y_pred
                total_error += abs(error)

                # Update weights and bias  5
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

            # Calculate metrics at the end of each epoch
            accuracy = accuracy_score(y, predictions)
            precision = precision_score(y, predictions)
            recall = recall_score(y, predictions)
            f1 = f1_score(y, predictions)

            self.metrics_per_epoch["accuracy"].append(accuracy)
            self.metrics_per_epoch["precision"].append(precision)
            self.metrics_per_epoch["recall"].append(recall)
            self.metrics_per_epoch["f1"].append(f1)

            if total_error == 0:
                break

    def evaluate(self, X, y):
        predictions = [self.predict(x) for x in X]
        return predictions

# Train the perceptron
perceptron = SingleLayerPerceptron(num_features=train_data.shape[1])
perceptron.train(train_data, train_target_binary)

# Evaluate on test data
test_predictions = perceptron.evaluate(test_data, test_target_binary)
print("Test Predictions:", test_predictions)
print("Ground Truth:", test_target_binary)

# Compute metrics for test data
cm = confusion_matrix(test_target_binary, test_predictions)
accuracy = accuracy_score(test_target_binary, test_predictions)
precision = precision_score(test_target_binary, test_predictions)
recall = recall_score(test_target_binary, test_predictions)
f1 = f1_score(test_target_binary, test_predictions)
roc_auc = roc_auc_score(test_target_binary, test_predictions)

print("\nConfusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC-AUC:", roc_auc)

# Plot performance metrics over epochs
metrics = perceptron.metrics_per_epoch

plt.figure(figsize=(10, 6))
plt.plot(metrics["accuracy"], label="Accuracy", marker="o")
plt.plot(metrics["precision"], label="Precision", marker="s")
plt.plot(metrics["recall"], label="Recall", marker="^")
plt.plot(metrics["f1"], label="F1-score", marker="d")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Performance Metrics over Training Epochs")
plt.legend()
plt.grid()
plt.savefig("performance_metrics_over_epochs.png")
plt.show()
