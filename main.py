from data_loader import load_normalized_data

class Knn():
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def _euclidean(self, point_a, point_b):
        squared_dist = 0
        for coord_a, coord_b in zip(point_a, point_b):
            squared_dist += (coord_a - coord_b) ** 2
        return squared_dist ** 0.5

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        self.X_train = [list(row) for row in X]
        self.y_train = list(y)

    def predict(self, X):
        predictions = []

        for sample in X:
            distances = []

            for train_point, train_label in zip(self.X_train, self.y_train):
                distance = self._euclidean(sample, train_point)
                distances.append((distance, train_label))
            distances.sort()
            k_nearest = distances[:self.k]

            vote_counts = {}
            for distance, label in k_nearest:
                vote_counts[label] = vote_counts.get(label, 0) + 1
            
            predicted_label = None
            highest_count = -(float('inf'))
            for label, count in vote_counts.items():
                if count > highest_count:
                    highest_count = count
                    predicted_label = label
            predictions.append(predicted_label)

        return predictions

    def evaluate(self, X, y):
        pass

    def grid_search(self, X_train, y_train, X_val, y_val, param_grid):
        pass