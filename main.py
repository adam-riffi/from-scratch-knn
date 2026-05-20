from data_loader import load_normalized_data

class Knn():
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def _euclidean(self, a, b):
        squared_sum = 0
        for x, y in zip(a, b):
            squared_sum += (x - y) ** 2
        return squared_sum ** 0.5

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass

    def grid_search(self, X_train, y_train, X_val, y_val, param_grid):
        pass