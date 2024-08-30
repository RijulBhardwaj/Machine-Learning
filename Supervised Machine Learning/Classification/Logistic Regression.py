import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate sample data
np.random.seed(0)
hours_studied = np.random.uniform(0, 10, 100)
marks = (hours_studied > 5).astype(int)

# Reshape the data
X = hours_studied.reshape(-1, 1)
y = marks

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.xlabel('Hours studied')
plt.ylabel('Pass (1) / Fail (0)')
plt.title('Logistic Regression: Study Hours vs. Exam Result')

# Plot the decision boundary
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
y_plot = model.predict_proba(X_plot)[:, 1]
plt.plot(X_plot, y_plot, color='red', label='Logistic Regression')

plt.legend()
plt.grid(True)
plt.show()
