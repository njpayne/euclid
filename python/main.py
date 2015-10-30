import scipy as sp
import numpy as np 
import math
import os
import csv

from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler #used to convert categories to one of k 
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import zero_one_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import classifiers, clustering

data_location = "../Data" # read data from os.path.join(data_location, <filename>)
results_location = "Results" # save results text/graph to os.path.join(results_location, <filename>)

def load_data(filename, conversion_function = None, max_records = None):
    """
    Loads the data from a csv

    Parameters
    ----------
        filename: name of csv file to be read in the data_location folder
    
    Returns
    -------
        (header, data): tuple of lists, a list of data headings and a list of data
    """
    
    header = []
    data = []

    #open the csv file
    with open(os.path.join(data_location, filename), 'rb') as csvfile:
        #read each line
        for line in csvfile.readlines():
            line_data = line.strip().split(',')
            #assume the first line is the header.  Add to the heading list
            if(len(header) == 0):
                header = line_data
            #otherwise add to the data list
            else:
                data.append(line_data)

    #just so we have a random set sort then subset
    if(max_records is not None and len(data) >= max_records):
        random.shuffle(data)
        data = data[ : max_records + 1]

    #convert to a numpy array
    header, data = np.array(header), np.array(data)

    #convert the data to categories
    if(conversion_function is not None):
        header, data = conversion_function(header, data)

    return (header, data)

def convert_survey_data(header, data):
    """
    Converts data from survey to usable form for analysis

    Parameters
    ----------
        data: the raw survey data
    
    Returns
    -------
        converted_data: the survey data after conversion      

    """

    #this list represents the indexes to be converted to a binary representation
    #category_indexes = [10,11,12,13,14,15,16,17,18,19,20,40,41,42,43,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,76,77,78,79,80,89,90,91,92,93,94,95,96,97,98,99]
    category_indexes = np.argwhere([not classifiers.is_number(data[0, i]) for i in range(data.shape[1])]).astype(np.int).flatten().tolist()

    #label encoder converts categories to numerical values
    label_encoder = LabelEncoder()
    
    #loop through all the columns that are categorical data and convert
    for i in category_indexes:
        label_encoder.fit(data[ : , i])
        data[ : , i] = label_encoder.fit_transform(data[ : , i])

    #convert the data to floats
    data = data.astype(np.float)

    #the OneHotEncoder converts categorical data to binary representations
    encoder = OneHotEncoder(categorical_features = category_indexes)

    #convert the data
    converted_data = encoder.fit_transform(data).toarray()
    
    #the one hot encoder puts new columns on the low end of the array, so lets put them back
    converted_columns = converted_data.shape[1] - data.shape[1] + len(category_indexes)
    converted_data = np.hstack((converted_data[ : , converted_columns :], converted_data[ : , : converted_columns]))

    #convert the header to represent the transformed data
    unconverted_indexes = np.array([i not in category_indexes for i in range(len(header))])
    converted_header = header[unconverted_indexes]

    #add the new category names to the header
    for i in category_indexes:
        for j in range(data[ : , i].astype(np.int).max()):
            if(j == 0):
                converted_header = np.append(converted_header, header[i])
            else:
                converted_header = np.append(converted_header, header[i] + str(j))

    return converted_header, converted_data

def select_data_columns(header, data, column_names = []):
    """
    Selects the columns from the data based on the column_name list
    If header is not found, no data is added

    Parameters
    ----------
        header: data's header list
        data: list of the data to be split
        column_names: names of columns to be returned, checked against header file
    
    Returns
    -------
        sliced_header: headers that matched the column_names list
        sliced_data: data that matched the column_names list
    """

    column_indexes = []

    #get the index of the columns that match
    for i in range(len(column_names)):
        for j in range(len(header)):
            if(header[j] == column_names[i]):
                column_indexes.append(j)

    sliced_header = np.take(header, column_indexes, axis = 0)
    sliced_data = np.take(data, column_indexes, axis = 1)

    return (sliced_header, sliced_data)

def divide_for_training(data):
    ##first use the category for training and use the rest as features except for period code
    ##select_columns = ["names", "of", "columns"]
    #select_columns = header 

    ##select the appropriate columns
    #selected_header, selected_data = select_data_columns(header, data, select_columns)

    #have scikit partition the data into training and test sets
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[ : , 1 : ], data[ : ,  : 1], test_size=0.15, random_state=0)

    #create the scaler the data based on the training data
    #this is used to scale values to 0 mean and unit variance
    data_scaler = StandardScaler().fit(X_train.astype(np.float32))

    #scale training and test set to mean 0 with unit variance
    X_train = data_scaler.transform(X_train.astype(np.float32))
    X_test = data_scaler.transform(X_test.astype(np.float32))

    return X_train, X_test, y_train, y_test

def run_classifiers(X_train, y_train, X_test, y_test, header):
    print("------------------")
    print("Running Classifiers")
    print("------------------")

    print("\n\n--------------------------")
    print("Decision Trees")
    print("--------------------------")

    #create decision tree range
    decision_tree_param = {'max_depth': range(1, 200, 10), 'criterion' : ["entropy", "gini"]}

    #run the decision tree
    prediction, decision_tree_accuracy = classifiers.run_decision_tree(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = decision_tree_param, headings = header)
    print("Decision tree accuracy = %f" % decision_tree_accuracy)


    print("\n\n--------------------------")
    print("Boosting")
    print("--------------------------")

    #create boosting range
    boosting_param = {'base_estimator__max_depth': range(1, 3), 'n_estimators' : [10, 20], 'learning_rate' : [.75, 1.0] }

    #run the boosting
    prediction, boosting_accuracy = classifiers.run_boosting(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = boosting_param)
    print("Boosting accuracy = %f" % boosting_accuracy)


    print("\n\n--------------------------")
    print("k - Nearest Neighbors")
    print("--------------------------")


    #create knn range
    knn_param = {'n_neighbors': range(1, 20), 'weights': ['uniform', 'distance'], 'p': [1, 2], 'algorithm' : ['auto'], 'metric': ['euclidean']} #, 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis']}

    #run the knn
    prediction, knn_accuracy = classifiers.run_k_nearest_neighbors(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = knn_param)
    print("k-NN accuracy = %f" % knn_accuracy)

    print("\n\n--------------------------")
    print("SVM")
    print("--------------------------")

    #create svm range
    svm_param = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'C': [1e0, 5e0, 1e1, 5e1, 1e2, 5e2], 'degree': [1, 2, 3, 4], 'gamma': [0.0, 0.0001, 0.0005, 0.001]}

    #run the svm
    prediction, svm_accuracy = classifiers.run_support_vector_machines(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = svm_param)
    print("SVM accuracy = %f" % svm_accuracy)

    #print("\n\n--------------------------")
    #print("Neural Net")
    #print("--------------------------")

    ##run the neural net
    #prediction, nn_accuracy = classifiers.run_neural_net(X_train, y_train.flatten(), X_test, y_test.flatten())
    #print("Neural Net accuracy = %f" % nn_accuracy)
    
    #return  max(decision_tree_accuracy, boosting_accuracy, knn_accuracy, svm_accuracy, nn_accuracy)
    return  max(decision_tree_accuracy, boosting_accuracy, knn_accuracy, svm_accuracy)

def run_regressors(X_train, y_train, X_test, y_test, header):



    return


def test_all_features(header, data):


    #remove low variance features
    cleaned_data = clustering.clean_features(np.vstack((X_train, X_test)))
    
    #select the best features using univatiate selection
    selected_features, feature_uni_scores = clustering.univariate_selection(cleaned_data, np.vstack((y_train, y_test)))

    #grab the best columns based on the univariate test
    #best_feature_index = np.argsort(-feature_uni_scores)[:20]
    best_feature_index = np.argsort(-feature_uni_scores)[:]

    #reselect the features
    top_features = 10
    X_train, X_test = np.take(X_train, best_feature_index[ : 10], axis = 1), np.take(X_test, best_feature_index[ : 10], axis = 1)
    best_classifier = run_classifiers(X_train, y_train, X_test, y_test, selected_header)

    #try the PCA reduction
    reduced_data = clustering.pca_reduce(cleaned_data, n_components = 2)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(reduced_data, np.vstack((y_train, y_test)), test_size=0.15, random_state=0)

    best_classifier = run_classifiers(X_train, y_train, X_test, y_test, selected_header)

    return

def feature_reduction(header, data, is_classification = True):

    #remove low variance features
    data_lvar_removed, header_lvar_removed = clustering.clean_features(data, header, min_feature_variance = .8 * (1 - .8))

    #create training/test sets
    X_train, X_test, y_train, y_test = divide_for_training(data_lvar_removed)

    #select the best features using univatiate selection
    selected_features, feature_uni_scores = clustering.univariate_selection(np.vstack((X_train, X_test)), np.vstack((y_train, y_test)), n_best = 3, is_regression = not is_classification)

    #determine the order of the univariate features
    best_feature_index = np.argsort(-feature_uni_scores)

    #reselect the features
    top_features = 10
    X_train_uni, X_test_uni = np.take(X_train, best_feature_index[ : top_features], axis = 1), np.take(X_test, best_feature_index[ : top_features], axis = 1)
    header_uni = np.take(header_lvar_removed, best_feature_index[ : top_features])

    #try the PCA reduction
    data_pca = clustering.pca_reduce(np.vstack((X_train, X_test)), n_components = top_features)
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = divide_for_training(np.hstack((np.vstack((y_train, y_test)), data_pca)))

    if(is_classification):
        #run the classifiers and find the best result
        best_classifier_uni = run_classifiers(X_train_uni, y_train, X_test_uni, y_test, header_uni)
        best_classifier_pca = run_classifiers(X_train_pca, y_train_pca, X_test_pca, y_test_pca, header_lvar_removed)
    else:
        best_regressor_uni = run_regressors(X_train_uni, y_train, X_test_uni, y_test, header_uni)
        best_regressor_pca = run_regressors(X_train_pca, y_train_pca, X_test_pca, y_test_pca, header_lvar_removed)

    return

def main():
    #load the data from the csv
    header, data = load_data("basic_data.csv", conversion_function = convert_survey_data, max_records = None)

    #remove the grades to set up for classification
    classification_header = np.take(header, [0] + range(10, len(header)))
    classification_data = np.take(data, [0] + range(10, len(header)), axis = 1)

    #remove the grades to set up for regression
    regression_header = np.take(header, [1] + range(10, len(header)))
    regression_data = np.take(data, [1] + range(10, len(header)), axis = 1)

    #run feature reduction
    feature_reduction(classification_header, classification_data, is_classification = True)

    #test all to start
    test_all_features(header, data)

    #loop through all features individually
    #individual_feature_testing(header, data)

    return

if __name__ == "__main__":
    main()