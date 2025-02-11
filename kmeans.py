import random
import math

class KMeans:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []
    
    def initialize_centroids(self, data):
        """Randomly initialize k centroids from the data points"""
        self.centroids = random.sample(data, self.k)
    
    def euclidean_distance(self, point1, point2):
        """Compute Euclidean distance between two points"""
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
    
    def assign_clusters(self, data):
        """Assign each data point to the nearest centroid"""
        clusters = [[] for _ in range(self.k)]
        for point in data:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)
        return clusters
    
    def compute_new_centroids(self, clusters):
        """Compute new centroids as the mean of each cluster"""
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
            else:
                new_centroid = random.choice(self.centroids)  # Avoid empty clusters
            new_centroids.append(new_centroid)
        return new_centroids
    
    def fit(self, data):
        """Run K-Means clustering on the dataset"""
        self.initialize_centroids(data)
        
        for _ in range(self.max_iters):
            clusters = self.assign_clusters(data)
            new_centroids = self.compute_new_centroids(clusters)
            
            if new_centroids == self.centroids:  # Convergence check
                break
            
            self.centroids = new_centroids
        
        return self.centroids, clusters

# Example usage
data = [
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
    [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0],
    [9.0, 3.0]
]

kmeans = KMeans(k=2)
centroids, clusters = kmeans.fit(data)

print("Final Centroids:", centroids)
print("Clusters:", clusters)
