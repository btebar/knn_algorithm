import numpy as np

class Evaluator:

    def get_cm(self, predictions, true_values, class_labels=None):
        if class_labels is None:
            class_labels = np.sort(np.unique(true_values))
        conf_matrix = np.zeros((len(class_labels), len(class_labels)))
        label_indices = dict()
        i = 0
        for c in class_labels:
            label_indices.update({c:i})
            i = i + 1
        
        for i in range(len(predictions)):
            p = label_indices.get(predictions[i], -1)
            t = label_indices.get(true_values[i][0], -1)
            conf_matrix[t][p] = conf_matrix[t][p] + 1
        
        return conf_matrix

    def accuracy(self, confusion_matrix):
        num = 0
        den = 0
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix)):
                if i == j:
                    num = num + confusion_matrix[i][j]
                den = den + confusion_matrix[i][j]
        if den == 0:
            return 0

        return num/den

    def precision(self, confusion_matrix):
        p = np.zeros(len(confusion_matrix))
        for i in range(len(confusion_matrix)):
            num = 0
            den = 0
            for j in range(len(confusion_matrix)):
                if i == j:
                    num = confusion_matrix[i][j]
                den = confusion_matrix[j][i]
            if den != 0:   
                p[i] = num/den

        return p, np.average(p)

    def recall(self, confusion_matrix):
        r = np.zeros(len(confusion_matrix))
        for i in range(len(confusion_matrix)):
            num = 0
            den = 0
            for j in range(len(confusion_matrix)):
                if i == j:
                    num = confusion_matrix[i][j]
                den = confusion_matrix[i][j]
            if den == 0:
                r[i] = 0
            else:
                r[i] = num/den

        return r, np.average(r)

    def f_measure(self, alpha):
        _, p = self.precision()
        _, r = self.recall()

        num = p * r * (1 + alpha**2)
        den = alpha**2 * p + r
        if den == 0:
            return 0
        return num/den