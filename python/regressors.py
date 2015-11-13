import numpy as np
import time
import pylab
import os
import math
import pydot
import matplotlib.pyplot as plt

from sklearn import tree, neighbors, svm, ensemble, linear_model, svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.learning_curve import learning_curve
from sklearn.externals.six import StringIO  

from sklearn.cross_validation import ShuffleSplit, cross_val_predict, cross_val_score
from sklearn.grid_search import GridSearchCV
from plot_learning_curve import plot_learning_curve, plot_validation_curve, plot_learning_curve_iter, plot_adaclassifier

data_location = "../Data" # read data from os.path.join(data_location, <filename>)
results_location = "Results" # save results text/graph to os.path.join(results_location, <filename>)

def run_decision_tree(training_features, training_labels, test_features, test_labels, passed_parameters = None, headings = None, title = ""):
    """
    Regresses the data using sklearn's decision tree 
    Does not natively support pruning so max_depth is being used

    Parameters
    ----------
        training_data: data used to train the classifier. For each row, item 0 assumed to be the label
        test_data: data used to test the regressor. 
    
    Returns
    -------
        prediction: predicted labels of the test data
        accuracy: percent of test data labels accurately predicted
    """

    estimator = tree.DecisionTreeRegressor()
    
    #set up parameters for the classifier
    if(passed_parameters == None):
        parameters = {'max_depth': None}
    else:
        parameters = passed_parameters

    #create cross validation iterator
    cv = ShuffleSplit(training_features.shape[0], n_iter=5, test_size=0.2, random_state=0)

    #set up tuning algorithm
    regressor = GridSearchCV(estimator=estimator, cv=cv, param_grid=parameters)

    #fit the classifier
    regressor.fit(training_features, training_labels)

    test_prediction = regressor.predict(test_features)
    test_accuracy = regressor.score(test_features, test_labels)

    #show the best result
    estimator = tree.DecisionTreeRegressor(max_depth = regressor.best_estimator_.max_depth)
    estimator.fit(training_features, training_labels)

    #save the visualization of the decision tree only use the top 5 levels for now
    if(headings.shape[0] == training_features.shape[1]):
        tree_data = StringIO() 
        tree.export_graphviz(estimator, out_file=tree_data, max_depth=regressor.best_estimator_.max_depth, feature_names=headings)
        graph = pydot.graph_from_dot_data(tree_data.getvalue()) 
        graph.write_pdf(os.path.join(results_location, "Decision Tree Model_%s.pdf" % title)) 

    time_2 = time.time()



    ## Plot the results
    #plt.figure()
    #plt.scatter(X, y, c="k", label="data")
    #plt.plot(X_test, y_1, c="g", label="max_depth=2", linewidth=2)
    #plt.plot(X_test, y_2, c="r", label="max_depth=5", linewidth=2)
    #plt.xlabel("data")
    #plt.ylabel("target")
    #plt.title("Decision Tree Regression")
    #plt.legend()
    #plt.show()

    return test_prediction, test_accuracy

def run_boosting(training_features, training_labels, test_features, test_labels, passed_parameters = None):
    """
    Classifies the data using sklearn's ADAboost
    Does not natively support pruning so max_depth is being used for the decision tree

    Parameters
    ----------
        training_data: data used to train the classifier. For each row, item 0 assumed to be the label
        test_data: data used to test the classifier. For each row, item 0 assumed to be the label
        max_depth: maximum tree depth to be applied (will simulate pruning)
    
    Returns
    -------
        prediction: predicted labels of the test data
        accuracy: percent of test data labels accurately predicted
    """
    time_1 = time.time()

    #set up underlying decision tree classifier
    base_regressor = tree.DecisionTreeRegressor()

    #set up the boosting method
    estimator = ensemble.AdaBoostRegressor(base_estimator = base_regressor)
    
    #set up parameters for the classifier
    passed_parameters = {'base_estimator__max_depth': range(1, 5), 'n_estimators' : range(10, 200, 50), 'learning_rate' : [1] }

    #create cross validation iterator
    cv = ShuffleSplit(training_features.shape[0], n_iter=5, test_size=0.2, random_state=0)

    #set up tuning algorithm
    regressor = GridSearchCV(estimator=estimator, cv=cv, param_grid=passed_parameters)

    #fit the classifier
    regressor.fit(training_features, training_labels)

    #get the prediction and accuracy of the test set
    test_prediction = regressor.predict(test_features)
    test_accuracy = regressor.score(test_features, test_labels)

    return test_prediction, test_accuracy

def run_random_forest(training_features, training_labels, test_features, test_labels, passed_parameters = None, ):

    estimator = ensemble.RandomForestRegressor(random_state=0, n_estimators=25)

    #set up parameters for the classifier
    if(passed_parameters == None):
        parameters = {'max_depth': None}
    else:
        parameters = passed_parameters

    #create cross validation iterator
    cv = ShuffleSplit(training_features.shape[0], n_iter=5, test_size=0.2, random_state=0)

    #set up tuning algorithm
    regressor = GridSearchCV(estimator=estimator, cv=cv, param_grid=parameters)

    #fit the classifier
    regressor.fit(training_features, training_labels)

    test_prediction = regressor.predict(test_features)
    test_accuracy = regressor.score(test_features, test_labels)

    time_2 = time.time()

    return test_prediction, test_accuracy

def run_linear_regression(training_features, training_labels, test_features, test_labels, passed_parameters = None, headings = ["Linear"]):
    

    #set up linear regressor
    estimator = linear_model.LinearRegression(fit_intercept = True)

    estimator.fit(training_features, training_labels)

    prediction = estimator.predict(X = test_features)
    score = estimator.score(X = test_features, y = test_labels)

    if(training_features.shape[1] == 1):

        fig, ax = plt.subplots()
        ax.scatter(training_labels, prediction)
        ax.plot([training_labels.min(), training_labels.max()], [training_labels.min(), training_labels.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        pylab.savefig(os.path.join(results_location, "Linear - " + headings[-1] + '.png'))

    return prediction, score

def run_support_vector_regressor(training_features, training_labels, test_features, test_labels, passed_parameters = None):
    
    estimator = svm.SVR()

    #set up parameters for the classifier
    if(passed_parameters == None):
        parameters = {'kernel': ['linear']}
    else:
        parameters = passed_parameters

    #create cross validation iterator
    cv = ShuffleSplit(training_features.shape[0], n_iter=5, test_size=0.2, random_state=0)

    #set up tuning algorithm
    regressor = GridSearchCV(estimator=estimator, cv=cv, param_grid=parameters)

    #fit the classifier
    regressor.fit(training_features, training_labels)

    test_prediction = regressor.predict(test_features)
    test_accuracy = regressor.score(test_features, test_labels)

    time_2 = time.time()

    return test_prediction, test_accuracy
