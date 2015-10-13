import numpy as np
import time
import pylab
import os

from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.learning_curve import learning_curve

from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from plot_learning_curve import plot_learning_curve, plot_validation_curve, plot_learning_curve_iter



data_location = "../Data" # read data from os.path.join(data_location, <filename>)
results_location = "Results" # save results text/graph to os.path.join(results_location, <filename>)

def run_decision_tree(training_features, training_labels, test_features, test_labels, passed_parameters = None):
    """
    Classifies the data using sklearn's decision tree 
    Does not natively support pruning so max_depth is being used

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

    estimator = tree.DecisionTreeClassifier()
    
    #set up parameters for the classifier
    if(passed_parameters == None):
        parameters = {'max_depth': None}
    else:
        parameters = passed_parameters

    #create cross validation iterator
    cv = ShuffleSplit(training_features.shape[0], n_iter=5, test_size=0.2, random_state=0)

    #plot the validation curves
    for param in parameters:
        if(is_number(parameters[param][0])):
            title = 'Validation Curves \n(Decision Tree)' 
            save_name = "Validation Curves - Decision Tree - %s.png" % param
            plot_validation_curve(estimator, training_features, training_labels, title, param, parameters[param], cv)
            pylab.savefig(os.path.join(results_location, save_name))

    #set up tuning algorithm
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=parameters)

    #fit the classifier
    classifier.fit(training_features, training_labels)

    test_prediction = classifier.predict(test_features)
    test_accuracy = classifier.score(test_features, test_labels)

    time_2 = time.time()

    #show the best result
    estimator = tree.DecisionTreeClassifier(max_depth = classifier.best_estimator_.max_depth, criterion = classifier.best_estimator_.criterion)
    
    #plot the learning curve
    title = 'Learning Curves \n(Decision Tree, max depth=%i)' %classifier.best_estimator_.max_depth
    plot_learning_curve(estimator, title, training_features, training_labels, cv=cv)
    pylab.savefig(os.path.join(results_location, 'Learning Curves - Decision Tree.png'))
    #plt.show()

    time_3 = time.time()

    #output time stats
    #time 1 -> time 2 is optimization time
    #time 2 -> time 3 is run for just one case
    print("Decision Tree Time Stats")
    print("Optimization Time -> %f" % (time_2 - time_1))
    print("Single Run Time -> %f" % (time_3 - time_2))

    #output classification report and confusion matrix
    print('\n\n----------------------------')
    print('Classification Report')
    print('----------------------------\n')
    print(classification_report(y_true = test_labels, y_pred = test_prediction))
    
    print('\n\n----------------------------')
    print('Confusion Matrix')
    print('----------------------------\n')
    print(confusion_matrix(y_true = test_labels, y_pred = test_prediction))

    return test_prediction, test_accuracy


def is_number(s):
    """
        Simple helper method to determine if the parameter is a number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False