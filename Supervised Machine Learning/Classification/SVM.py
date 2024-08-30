import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a simple dataset: student exam scores
# Feature 1: Hours studied
# Feature 2: Previous test score
# Class: 0 for fail, 1 for pass
X = np.array([[2, 50], [4, 70], [3, 60], [5, 80], [6, 75], 
              [7, 85], [3, 55], [8, 90], [5, 65], [6, 78]])
y = np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the SVM classifier
def svm_classifier(X_train, y_train, X_test):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

# Make predictions
y_pred = svm_classifier(X_train, y_train, X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], color='red', label='Fail')
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], color='green', label='Pass')
plt.xlabel('Hours studied')
plt.ylabel('Previous test score')
plt.title('SVM: Student Exam Performance')

# Plot the decision boundary
clf = svm.SVC(kernel='linear')
clf.fit(X, y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)

plt.legend()
plt.grid(True)
plt.show()
