import numpy as np
import matplotlib.pyplot as plt

# Simple dataset: Height and Weight
X = np.array([[170, 70], [175, 75], [180, 80], [160, 60], [165, 65],
              [185, 85], [190, 90], [155, 55], [160, 58], [170, 68]])

def initialize_parameters(K, d):
    means = np.random.rand(K, d)
    covariances = np.array([np.eye(d) for _ in range(K)])
    weights = np.ones(K) / K
    return means, covariances, weights

def gaussian_pdf(x, mean, cov):
    d = mean.shape[0]
    diff = x - mean
    return np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff) / np.sqrt((2 * np.pi)**d * np.linalg.det(cov))

def expectation_step(X, means, covariances, weights):
    N, K = len(X), len(weights)
    responsibilities = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            responsibilities[i, k] = weights[k] * gaussian_pdf(X[i], means[k], covariances[k])
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

def maximization_step(X, responsibilities):
    N, K = responsibilities.shape
    d = X.shape[1]
    
    Nk = responsibilities.sum(axis=0)
    means = (responsibilities.T @ X) / Nk[:, np.newaxis]
    
    covariances = np.zeros((K, d, d))
    for k in range(K):
        diff = X - means[k]
        covariances[k] = (responsibilities[:, k, np.newaxis, np.newaxis] * diff[:, :, np.newaxis] @ diff[:, np.newaxis, :]).sum(axis=0) / Nk[k]
    
    weights = Nk / N
    return means, covariances, weights

def gaussian_mixture(X, K, max_iters=100):
    d = X.shape[1]
    means, covariances, weights = initialize_parameters(K, d)
    
    for _ in range(max_iters):
        responsibilities = expectation_step(X, means, covariances, weights)
        means, covariances, weights = maximization_step(X, responsibilities)
    
    return means, covariances, weights

# Run Gaussian Mixture
K = 2
means, covariances, weights = gaussian_mixture(X, K)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Data points')
for k in range(K):
    mean = means[k]
    cov = covariances[k]
    v, w = np.linalg.eigh(cov)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi
    ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[1], 180. + angle, color='red', alpha=0.3)
    plt.gca().add_artist(ell)
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Gaussian Mixture Model: Height vs Weight')
plt.legend()
plt.grid(True)
plt.show()

print("Cluster means:")
print(means)
print("\nCluster weights:")
print(weights)
