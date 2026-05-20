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

        y_true = list(y)
        y_pred = self.predict(X)
        n_samples = len(y_true)

        # Accuracy
        correct = -- (float('inf'))
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label == pred_label:
                correct += 1
        accuracy = correct / n_samples

        # Weighted precision, recall, F1-score
        classes = set(y_true)
        weighted_precision = 0
        weighted_recall = 0
        weighted_f1 = 0

        for current_class in classes:
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            support = 0

            for true_label, predicted_label in zip(y_true, y_pred):
                if true_label == current_class:
                    support += 1
                    if predicted_label == current_class:
                        true_positives += 1
                    else:
                        false_negatives += 1
                elif predicted_label == current_class:
                    false_positives += 1

            # Per-class metrics with division-by-zero safety
            if true_positives + false_positives == 0:
                precision = 0.0
            else:
                precision = true_positives / (true_positives + false_positives)

            if true_positives + false_negatives == 0:
                recall = 0.0
            else:
                recall = true_positives / (true_positives + false_negatives)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            # Weight by support (count of true samples of this class)
            weight = support / n_samples
            weighted_precision += precision * weight
            weighted_recall += recall * weight
            weighted_f1 += f1 * weight

        return {
            "accuracy": accuracy,
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        } 





    def grid_search(self, X_train, y_train, X_val, y_val, param_grid):
        pass