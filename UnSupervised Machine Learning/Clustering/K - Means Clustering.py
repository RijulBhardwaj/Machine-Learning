# K-Means Clustering

# Define a simple dataset
data = [
    [2, 3], [2, 5], [1, 2], [5, 8], [7, 9],
    [8, 10], [9, 5], [10, 8], [10, 10], [11, 7]
]

# Initialize centroids
centroids = [[2, 3], [8, 10]]

def euclidean_distance(point1, point2):
    return sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5

def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
    return clusters

def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if cluster:
            centroid = [sum(coord) / len(cluster) for coord in zip(*cluster)]
            new_centroids.append(centroid)
    return new_centroids

def k_means(data, k, max_iterations=100):
    centroids = data[:k]
    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return clusters, centroids

# Run K-Means
final_clusters, final_centroids = k_means(data, 2)

# Print results
print("Final Clusters:")
for i, cluster in enumerate(final_clusters):
    print(f"Cluster {i + 1}: {cluster}")

print("\nFinal Centroids:")
for i, centroid in enumerate(final_centroids):
    print(f"Centroid {i + 1}: {centroid}")

# Plot the results
x = [point[0] for point in data]
y = [point[1] for point in data]

print("\nPlot:")
print("x y")
for i in range(len(x)):
    print(f"{x[i]} {y[i]}")

print("\nTo visualize the results, you can copy the x and y coordinates")
print("into a plotting tool or spreadsheet to create a scatter plot.")
print("The points in each cluster will be grouped together,")
print("and the centroids will be at the center of each cluster.")
