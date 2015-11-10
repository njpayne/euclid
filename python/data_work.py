import math
import os
import csv
import numpy as np

from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler #used to convert categories to one of k 

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
    category_indexes = np.argwhere([not is_number(data[0, i]) for i in range(data.shape[1])]).astype(np.int).flatten().tolist()

    #label encoder converts categories to numerical values
    label_encoder = LabelEncoder()
    
    #loop through all the columns that are categorical data and convert
    encoder_indexes = []
    for i in category_indexes:
        label_encoder.fit(data[ : , i])
        encoder_indexes.append(label_encoder.classes_)
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
        #get the encoding labels
        category_labels = encoder_indexes[np.argwhere(np.array(category_indexes) == i)]
        for j in range(data[ : , i].astype(np.int).max() + 1):
            converted_header = np.append(converted_header, np.array([header[i] + "_" + category_labels[j]]))

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

def scale_features(data, test_data = None):

    #create the scaler the data based on the training data
    #this is used to scale values to 0 mean and unit variance
    data_scaler = StandardScaler().fit(data.astype(np.float32))
    scaled_data = data_scaler.transform(data)

    if(test_data is None):
        return scaled_data
    else:
        scaled_test_data = data_scaler.transform(test_data)    

    return scaled_data, scaled_test_data

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

def is_number(s):
    """
        Simple helper method to determine if the parameter is a number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False