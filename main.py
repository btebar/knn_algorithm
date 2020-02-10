import numpy as np
from knn_algo import KnnAlgorithm

if __name__ == "__main__":
    print("---- Loading Dataset... ----")
    data = np.loadtxt('iris.dat')
    rows = int(data.shape[0])
    
    
    sections = [int(rows - rows/15), rows]
    data = np.split(data, sections)
    training_data = data[0]
    
    data = data[1]
    columns = int(data.shape[1])

    sections = [int(columns-1), columns]
    testing_data = np.hsplit(data, sections)
    ground_truth = testing_data[1]
    testing_data = testing_data[0]
    
    print("----- Predicting... -----")
    algo = KnnAlgorithm()
    predictions = algo.predict_multiple(5, data, testing_data)
    for i in range(len(predictions)):
        if predictions[i] != ground_truth[i]:
            print("Item at position " + str(i) + " didn't generate correct output:")
            print("Ground truth: " + str(ground_truth[i] + ", prediction: " + str(predictions[i])))
        
    print("---- DONE! ----")


    # To fix: need to classify in three classes, last 
    # three columns are the different classes
    # 0, 1, 0 -> second class
    # and fix the way we return from get_k_neighbours