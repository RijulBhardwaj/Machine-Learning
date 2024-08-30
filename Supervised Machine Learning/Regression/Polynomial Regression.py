import numpy as np
import matplotlib.pyplot as plt

# Simple dataset: Years of experience vs. Salary
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([30000, 35000, 42000, 50000, 65000, 72000, 85000, 92000, 105000, 120000])

# Normalize the features
X = (X - np.mean(X)) / np.std(X)

def hypothesis(X, theta):
    return theta[0] + theta[1] * X + theta[2] * X**2

def cost_function(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    
    for _ in range(iterations):
        predictions = hypothesis(X, theta)
        theta[0] = theta[0] - (alpha / m) * np.sum(predictions - y)
        theta[1] = theta[1] - (alpha / m) * np.sum((predictions - y) * X)
        theta[2] = theta[2] - (alpha / m) * np.sum((predictions - y) * X**2)
        cost_history.append(cost_function(X, y, theta))
    
    return theta, cost_history

# Initialize parameters
theta = np.zeros(3)
alpha = 0.01
iterations = 1000

# Run gradient descent
theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

# Generate predictions
X_pred = np.linspace(X.min(), X.max(), 100)
y_pred = hypothesis(X_pred, theta)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X_pred, y_pred, color='red', label='Polynomial Regression')
plt.xlabel('Years of experience (normalized)')
plt.ylabel('Salary')
plt.title('Polynomial Regression: Years of Experience vs. Salary')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Print the final parameters
print(f"Final parameters: theta0 = {theta[0]:.2f}, theta1 = {theta[1]:.2f}, theta2 = {theta[2]:.2f}")

# Print the final cost
print(f"Final cost: {cost_history[-1]:.2f}")
