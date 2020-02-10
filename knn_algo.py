import numpy as np
import math

class KnnAlgorithm:

    def euclidean_distance(self, data_row, item, num_attr):
        distance = 0
        for i in range(0, num_attr):
            distance = distance + (item[i] - data_row[i])**2
        distance = math.sqrt(distance)
        return distance

    def get_k_neighbours(self, k, dataset, item):
        num_attributes = len(item) - 1
        class_distances = dict()
        for i in range(0, dataset.shape[0]):
            distance = self.euclidean_distance(dataset[i], item, num_attributes)
            class_distances.update({item[num_attributes]: distance})

        class_distances = sorted(class_distances.items(), key=lambda kv: kv[1])    
        print(class_distances)
        return k_neighbours

    def calculate_weight(self, item):
        x, distance = item
        weight = float(1)/float(distance)
        return x, weight

    def predict(self, k, dataset, item):  
        k_neighbours = self.get_k_neighbours(k, dataset, item)
        votes = dict()
        for i in k_neighbours:
            # weight
            label, weight = self.calculate_weight(i)
            # sum weights of same class
            votes[label] = votes.get(label, 0) + 1
        # return class with max weight
        votes = sorted(votes.items(), key=lambda kv: kv[1])
        return votes[0][0]

    def predict_multiple(self, k, dataset, items):
        predictions = np.zeros(len(items))
        i = 0
        for item in items:
            predictions[i] = self.predict(k, dataset, item)
            i = i + 1
        return predictions



