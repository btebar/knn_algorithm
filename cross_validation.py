import numpy as np
import math as m
from knn_algo import KnnAlgorithm
from evaluation import Evaluator

class Cross_Validation:

    def __init__(self):
        self.knn_algo = KnnAlgorithm()
        self.evaluator = Evaluator()

    def cross_validation(self, dataset, k, params):
        fold_size = m.floor(len(dataset)/k)
        best_param = 1
        max_a = 0
        for i in range(k):
            folds = np.split(dataset, [i*fold_size, i*fold_size+fold_size, len(dataset)])
            test = folds[1]
            training = np.concatenate((folds[0], folds[2]))
            curr_p, acc = self.parameter_tuning(training, params, k-1)
            if max_a < acc:
                max_a = acc
                best_param = curr_p
        print(max_a)
        return best_param
    
    def parameter_tuning(self, training, params, k):
        fold_size = m.floor(len(training)/k)
        max_a = 0
        best_param = 1
        for i in range(k):
            folds = np.split(training, [i*fold_size, i*fold_size+fold_size, len(training)])
            validation_data = folds[1]
            training_data = np.concatenate((folds[0], folds[2]))
            columns = int(validation_data.shape[1])

            sections = [int(columns-1), columns]
            val_data = np.hsplit(validation_data, sections)
            ground_truth = val_data[1]
            val_data = val_data[0]
            for param in params:
                pred = self.knn_algo.predict_multiple(param, training_data, val_data)
                cm = self.evaluator.get_cm(pred, ground_truth)
                accuracy = self.evaluator.accuracy(cm)
                if accuracy > max_a:
                    max_a = accuracy
                    best_param = param

        return best_param, max_a
            
            


