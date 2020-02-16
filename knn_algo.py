import numpy as np
import math

class KnnAlgorithm:

    def get_classes(self, items):
        label_position = items.shape[1]-3
        classes = [1, 2, 3]
        new_items = np.zeros(shape=(items.shape[0], label_position+1))
        for k in range(0, len(items)):
            item = items[k]
            i = 0
            while item[label_position + i] == 0:
                i = i + 1
            for j in range(0, new_items.shape[1]-1):
                new_items[k][j] = item[j]
            new_items[k][new_items.shape[1]-1] = classes[i]
        return new_items

    def euclidean_distance(self, data_row, item, num_attr):
        distance = 0
        for i in range(0, num_attr):
            distance = distance + (item[i] - data_row[i])**2
        distance = math.sqrt(distance)
        return distance

    def get_k_neighbours(self, k, dataset, item):
        num_attributes = len(item)
        class_distances = list()
        k_neighbours = list()
        for i in range(0, dataset.shape[0]):
            distance = self.euclidean_distance(dataset[i], item, num_attributes)
            class_distances.append((dataset[i][num_attributes], distance))

        class_distances.sort(key=lambda tup: tup[1])
        k_neighbours = class_distances[:k]
        return k_neighbours

    def calculate_weight(self, item):
        x, distance = item
        if distance == 0:
            return x, 1
        weight = float(1)/float(distance)
        return x, weight

    def predict(self, k, dataset, item): 
        k_neighbours = self.get_k_neighbours(k, dataset, item)
        votes = dict()
        for i in k_neighbours:
            # weight
            label, weight = self.calculate_weight(i)
            # sum weights of same class
            votes[label] = votes.get(label, 0) + weight
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



