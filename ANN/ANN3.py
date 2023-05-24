# Adaptive Resonance Theory(ART)

import numpy as np

class ART1:
    def __init__(self, num_features, vigilance):
        self.num_features = num_features
        self.vigilance = vigilance
        self.num_clusters = 0
        self.weights = np.zeros((1, num_features))
        self.categories = []

    def train(self, input_pattern):
        while True:
            for i in range(self.num_clusters):
                match = self.compute_matching(input_pattern, self.weights[i])
                if match >= self.vigilance:
                    self.update_weights(input_pattern, self.weights[i])
                    return self.categories[i]

            self.num_clusters += 1
            self.weights = np.vstack((self.weights, input_pattern))
            self.categories.append(self.num_clusters)
            return self.num_clusters

    def compute_matching(self, input_pattern, weight):
        return np.sum(np.minimum(input_pattern, weight)) / np.sum(input_pattern)

    def update_weights(self, input_pattern, weight):
        weight += input_pattern
        weight[weight < 1] = 0


input_patterns = np.array([
    [0, 1, 0, 1, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1]
])

art = ART1(num_features=len(input_patterns[0]), vigilance=0.5)

for pattern in input_patterns:
    cluster = art.train(pattern)
    print(f"Pattern: {pattern} | Cluster: {cluster}")