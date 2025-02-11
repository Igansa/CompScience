class KNN:
    def __init__(self, k):
        self.k = k
        self.data = []
    
    def euclidean_distance(self, point1, point2):
        """Compute Euclidean distance between two points"""
        sum_sq = 0
        for p1, p2 in zip(point1, point2):
            sum_sq += (p1 - p2) * (p1 - p2)
        return sum_sq ** 0.5
    
    def fit(self, data, labels):
        """Store the training data and labels"""
        self.data = []
        for i in range(len(data)):
            self.data.append((data[i], labels[i]))
    
    def predict(self, points):
        """Predict labels for a list of points"""
        predictions = []
        for point in points:
            distances = []
            for data_point, label in self.data:
                distances.append((self.euclidean_distance(point, data_point), label))
            
            # Sort distances manually
            for i in range(len(distances)):
                for j in range(i + 1, len(distances)):
                    if distances[i][0] > distances[j][0]:
                        distances[i], distances[j] = distances[j], distances[i]
            
            nearest_labels = []
            for i in range(self.k):
                nearest_labels.append(distances[i][1])
            
            # Find most common label manually
            label_counts = {}
            for label in nearest_labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
            
            most_common = None
            max_count = 0
            for label, count in label_counts.items():
                if count > max_count:
                    most_common = label
                    max_count = count
            
            predictions.append(most_common)
        return predictions

# Example usage
data = [
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
    [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0],
    [9.0, 3.0], [3.0, 3.0], [2.5, 2.7], [7.0, 7.5]
]
labels = ['A', 'A', 'B', 'B', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'B']

test_points = [[2.0, 2.0], [6.0, 6.0], [3.5, 3.5], [8.5, 8.5]]

knn = KNN(k=3)
knn.fit(data, labels)
predictions = knn.predict(test_points)

print("Predictions:", predictions)
