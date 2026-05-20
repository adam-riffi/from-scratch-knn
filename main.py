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
        pass

    def evaluate(self, X, y):
        pass

    def grid_search(self, X_train, y_train, X_val, y_val, param_grid):
        pass