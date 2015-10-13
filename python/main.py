﻿import scipy as sp
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

import classifiers

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
    category_indexes = [1] + range(5, 18, 1) + range(19, 22, 1)

    label_encoder = LabelEncoder()
    
    for i in category_indexes:
        label_encoder.fit(data[ : , i])
        data[ : , i] = label_encoder.fit_transform(data[ : , i])

    #the OneHotEncoder converts categorical data to binary representations
    #encoder = OneHotEncoder(categorical_features = category_indexes)

    #TODO convert the data
    converted_data = data
    #converted_data = encoder.transform(data).toarray()

    #TODO convert the header
    converted_header = header

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

def main():

    #load the data from the csv
    header, data = load_data("sample_survey_data.csv", conversion_function = convert_survey_data, max_records = None)

    #first use the category for training and use the rest as features except for period code
    #select_columns = ["names", "of", "columns"]
    select_columns = header

    #select the appropriate columns
    selected_header, selected_data = select_data_columns(header, data, select_columns)

    #have scikit partition the data into training and test sets
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(selected_data[ : , 1 : ], selected_data[ : ,  : 1], test_size=0.2, random_state=0)

    #create the scaler the data based on the training data
    #this is used to scale values to 0 mean and unit variance
    data_scaler = StandardScaler().fit(X_train.astype(np.float32))

    #scale training and test set to mean 0 with unit variance
    X_train = data_scaler.transform(X_train.astype(np.float32))
    X_test = data_scaler.transform(X_test.astype(np.float32))

    print("------------------")
    print("Running Classifiers")
    print("------------------")

    print("\n\n--------------------------")
    print("Decision Trees")
    print("--------------------------")

    #create decision tree range
    decision_tree_param = {'max_depth': range(1, 200, 10), 'criterion' : ["entropy", "gini"]}

    #run the decision tree
    prediction, accuracy = classifiers.run_decision_tree(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = decision_tree_param)
    print("Decision tree accuracy = %f" % accuracy)




    return

if __name__ == "__main__":
    main()