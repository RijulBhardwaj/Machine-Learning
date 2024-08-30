import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define a simple dataset: fruit classification based on weight and color
# Color: 0 for red, 1 for yellow
# Weight in grams
# Class: 0 for apple, 1 for banana
X = np.array([[150, 0], [170, 0], [140, 0], [130, 0],
              [180, 1], [160, 1], [170, 1], [190, 1]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the Naive Bayes classifier
def naive_bayes_classifier(X_train, y_train, X_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model.predict(X_test)

# Make predictions
y_pred = naive_bayes_classifier(X_train, y_train, X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], color='red', label='Apple')
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], color='yellow', label='Banana')
plt.xlabel('Weight (grams)')
plt.ylabel('Color (0: Red, 1: Yellow)')
plt.title('Naive Bayes: Fruit Classification')

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 0.01))
Z = naive_bayes_classifier(X_train, y_train, np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)

plt.legend()
plt.grid(True)
plt.show()
