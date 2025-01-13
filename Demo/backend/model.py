import math
from collections import Counter
import numpy as np

class KNNModel:
    def __init__(self, k=55):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, point1, point2):
        distance = 0
        for i in range(len(point1)):
            distance += (point1[i] - point2[i]) ** 2
        return math.sqrt(distance)

    def predict_single(self, test_point):
        distances = []
        # Tính khoảng cách từ test_point đến tất cả các điểm trong X_train
        for i in range(len(self.X_train)):
            distance = self.euclidean_distance(self.X_train[i], test_point)
            distances.append((distance, self.y_train[i]))

        # Sắp xếp theo khoảng cách tăng dần và lấy k điểm gần nhất
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = [label for _, label in distances[:self.k]]

        # Đếm số lượng nhãn trong k hàng xóm gần nhất
        most_common = Counter(k_nearest_neighbors).most_common(1)
        return most_common[0][0]

    def predict(self, X_test):
        y_pred = []
        for test_point in X_test:
            prediction = self.predict_single(test_point)
            y_pred.append(prediction)
        return y_pred