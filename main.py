import numpy as np
from knn_algo import KnnAlgorithm
from evaluation import Evaluator
from cross_validation import Cross_Validation

if __name__ == "__main__":
    
    algo = KnnAlgorithm()
    print("---- Loading Dataset... ----")
    data = np.loadtxt('iris.dat')
    data = algo.get_classes(data)
    # shuffle data
    np.random.shuffle(data)
    params = np.arange(1,15, dtype=int)
    cross_val = Cross_Validation()
    print("----- CROSS VALIDATION -----")
    k = cross_val.cross_validation(data, 10, params)
    print("Parameter chosen is: " + str(k))
    rows = int(data.shape[0])
    sections = [int(rows - rows/5), rows]
    data = np.split(data, sections)
    training_data = data[0]
    
    data = data[1]
    columns = int(data.shape[1])

    sections = [int(columns-1), columns]
    testing_data = np.hsplit(data, sections)
    ground_truth = testing_data[1]
    testing_data = testing_data[0]
    
    print("----- Predicting... -----")

    predictions = algo.predict_multiple(k, training_data, testing_data)
    num_errors = 0
    for i in range(len(predictions)):
        if predictions[i] != ground_truth[i]:
            num_errors = num_errors + 1
            print("Item at position " + str(i) + " didn't generate correct output:")
            print("Ground truth: " + str(ground_truth[i][0]) + ", prediction: " + str(predictions[i]))
    
    print("---- DONE! ----")
    print("Num errors: " + str(num_errors) + " total of predictions: " + str(len(predictions)))