import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a simple dataset: fruit classification
# Feature 1: Weight (grams)
# Feature 2: Sweetness level (1-10)
# Class: 0 for apple, 1 for orange
X = np.array([[150, 7], [130, 6], [180, 8], [160, 7], 
              [140, 5], [170, 8], [120, 4], [190, 9]])
y = np.array([0, 0, 1, 0, 0, 1, 0, 1])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the decision tree classifier
def tree_classifier(X_train, y_train, X_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

# Make predictions
y_pred = tree_classifier(X_train, y_train, X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], color='red', label='Apple')
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], color='orange', label='Orange')
plt.xlabel('Weight (grams)')
plt.ylabel('Sweetness level')
plt.title('Decision Tree: Fruit Classification')

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 0.1))
Z = tree_classifier(X_train, y_train, np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)

plt.legend()
plt.grid(True)
plt.show()
