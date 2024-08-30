import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define a simple dataset
hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
marks = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 0: Fail, 1: Pass

# Reshape the data
X = hours_studied.reshape(-1, 1)
y = marks

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the KNN classifier
def knn_classifier(X_train, y_train, X_test, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

# Make predictions
y_pred = knn_classifier(X_train, y_train, X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.xlabel('Hours studied')
plt.ylabel('Pass (1) / Fail (0)')
plt.title('KNN: Study Hours vs. Exam Result')

# Plot the decision boundary
X_plot = np.linspace(0, 11, 100).reshape(-1, 1)
y_plot = knn_classifier(X_train, y_train, X_plot)
plt.plot(X_plot, y_plot, color='red', label='KNN Decision Boundary')

plt.legend()
plt.grid(True)
plt.show()
