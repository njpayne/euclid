import numpy as np
import time
import pylab
import os

from sklearn import tree, neighbors, svm
from sklearn.metrics import classification_report, confusion_matrix
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

def run_neural_net(training_features, training_labels, test_features, test_labels):
    """
    Classifies the data using pybrain's neural net

    Parameters
    ----------
        training_data: data used to train the classifier. For each row, item 0 assumed to be the label
        test_data: data used to test the classifier. For each row, item 0 assumed to be the label
        hidden_units: sets the hidden unit count for the neural net
        training_epochs: sets the training epochs for the neural net
        training_iterations: # of training loops 
    
    Returns
    -------
        prediction: predicted labels of the test data
        accuracy: percent of test data labels accurately predicted
    """
    
    
    time_1 = time.time()

    #set the number of classes in the data
    number_of_outputs = training_labels.astype(int).max() + 1
    number_of_inputs = training_features.shape[1]

    #determine optimal hidden nodes based on Huang et al. (2003)
    first_layer_nodes = int(math.sqrt((number_of_outputs + 2) * number_of_inputs) + 2 * math.sqrt(number_of_inputs / (number_of_outputs + 2)))
    second_layer_nodes = int(number_of_outputs * math.sqrt(number_of_inputs / (number_of_outputs + 2)))

    #set up the layers
    input_layer = mlp_nn.Layer("Linear", units = number_of_inputs)
    hidden_layer1 = mlp_nn.Layer("Sigmoid", units = first_layer_nodes)
    hidden_layer2 = mlp_nn.Layer("Sigmoid", units = second_layer_nodes)
    output_layer = mlp_nn.Layer("Softmax", units = number_of_outputs)
    layers = [input_layer, hidden_layer1, hidden_layer2, output_layer]

    #set up the classifier
    neural_net = mlp_nn.Classifier(layers = layers, learning_rate = 0.02, n_iter = 5)

    #set up tuning parameters
    parameters = {"learning_rate" : [0.02], "n_iter" : [1, 5, 10, 25, 50]}

    #create cross validation iterator
    cv = ShuffleSplit(training_features.shape[0], n_iter=5, test_size=0.2, random_state=0)

    #set up tuning algorithm
    classifier = GridSearchCV(estimator=neural_net, cv=cv, param_grid=parameters)

    classifier.fit(training_features, training_labels)

    test_prediction = classifier.predict(test_features)
    test_accuracy = classifier.score(test_features, test_labels)

    time_2 = time.time()
    
    graph_title = "Learning Curves \n(Neural Net, learning rate=%f)" % classifier.best_estimator_.learning_rate
    plot_learning_curve_iter(classifier, graph_title)
    pylab.savefig(os.path.join(results_location, 'Validator Curves - Neural Net.png'))


    time_3 = time.time()

    #output time stats
    #time 1 -> time 2 is optimization time
    #time 2 -> time 3 is run for just one case
    print("Neural Net Time Stats")
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

def run_support_vector_machines(training_features, training_labels, test_features, test_labels, passed_parameters = None):
    """
    Classifies the data using sklearn's support vector machine classifier

    Parameters
    ----------
        training_data: data used to train the classifier. For each row, item 0 assumed to be the label
        test_data: data used to test the classifier. For each row, item 0 assumed to be the label
        kernel: (optional) Kernel to be used in the svm classifier can be 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    
    Returns
    -------
        prediction: predicted labels of the test data
        accuracy: percent of test data labels accurately predicted
    """

    time_1 = time.time()

    estimator = svm.SVC()
    
    #set up parameters that will be used by all kernels
    if(passed_parameters is None):
        parameters = {'C': [1e0, 5e0, 1e1, 5e1]}
    else:
        parameters = passed_parameters 

    #create cross validation iterator
    cv = ShuffleSplit(training_features.shape[0], n_iter=5, test_size=0.2, random_state=0)

    #plot the validation curves
    for param in parameters:
        if(is_number(parameters[param][0])):
            title = 'Validation Curves'
            save_name = "Validation Curves - SVC - %s.png" % param
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
    estimator = svm.SVC(kernel = classifier.best_estimator_.kernel, C = classifier.best_estimator_.C, gamma = classifier.best_estimator_.gamma, degree = classifier.best_estimator_.degree)

    #plot the learning curve
    title = 'Learning Curves (SVM, kernel=%s degree=%i gamma=%f C=%i )' % (classifier.best_estimator_.kernel, classifier.best_estimator_.degree, classifier.best_estimator_.gamma, classifier.best_estimator_.C)
    plot_learning_curve(estimator, title, training_features, training_labels, cv=cv)
    save_file_name = 'Learning Curves - SVM.png'
    pylab.savefig(os.path.join(results_location, save_file_name))
    #plt.show()

    time_3 = time.time()

    if(classifier.best_estimator_.kernel == 'linear'):
        coefficients = classifier.estimator.coef_
        print('\n\n-----------------------')
        print(' Coefficients')
        print(coefficients)

    #output time stats
    #time 1 -> time 2 is optimization time
    #time 2 -> time 3 is run for just one case
    print("SVM Time Stats")
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

def run_k_nearest_neighbors(training_features, training_labels, test_features, test_labels, passed_parameters = None):
    """
    Classifies the data using sklearn's k nearest neighbors classifier

    Parameters
    ----------
        training_data: data used to train the classifier. For each row, item 0 assumed to be the label
        test_data: data used to test the classifier. For each row, item 0 assumed to be the label
        k: number of nearest neighbors used in the algorithm
    
    Returns
    -------
        prediction: predicted labels of the test data
        accuracy: percent of test data labels accurately predicted
    """

    time_1 = time.time()

    estimator = neighbors.KNeighborsClassifier()
    
    #set up parameters for the classifier
    if(passed_parameters is None):
        parameters = {'n_neighbors': range(1, 11), 'weights': ['uniform', 'distance'], 'p': [1, 2] }
    else:
        parameters = passed_parameters

    #create cross validation iterator
    cv = ShuffleSplit(training_features.shape[0], n_iter=5, test_size=0.2, random_state=0)

    #plot the validation curves
    for param in parameters:
        if(is_number(parameters[param][0])):
            title = 'Validation Curves \n(kNN)' 
            save_name = "Validation Curves - kNN - %s.png" % param
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
    estimator = neighbors.KNeighborsClassifier(n_neighbors = classifier.best_estimator_.n_neighbors, weights = classifier.best_estimator_.weights, algorithm = classifier.best_estimator_.algorithm, leaf_size = classifier.best_estimator_.leaf_size, p = classifier.best_estimator_.p, metric = classifier.best_estimator_.metric)

    #plot the learning curve
    title = 'Learning Curves \n(k-NN, k-neighbors=%i weights=%s algorithm=%s leaf size=%i p=%i )' % (classifier.best_estimator_.n_neighbors, classifier.best_estimator_.weights, classifier.best_estimator_.algorithm, classifier.best_estimator_.leaf_size, classifier.best_estimator_.p)
    plot_learning_curve(estimator, title, training_features, training_labels, cv=cv)
    pylab.savefig(os.path.join(results_location, 'Learning Curves - kNN.png'))
    #plt.show()

    time_3 = time.time()

    #output time stats
    #time 1 -> time 2 is optimization time
    #time 2 -> time 3 is run for just one case
    print("kNN Time Stats")
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
    base_classifier = tree.DecisionTreeClassifier()

    #set up the boosting method
    estimator = ensemble.AdaBoostClassifier(base_estimator = base_classifier)
    
    #set up parameters for the classifier
    parameters = {'base_estimator__max_depth': range(1, 5), 'n_estimators' : range(10, 500, 50), 'learning_rate' : [.25, .5, .75, 1.0] }

    #create cross validation iterator
    cv = ShuffleSplit(training_features.shape[0], n_iter=5, test_size=0.2, random_state=0)

    #plot the validation curves
    for param in parameters:
        if(is_number(parameters[param][0])):
            title = 'Validation Curves \n(AdaBoost)' 
            save_name = "Validation Curves - AdaBoost - %s.png" % param
            plot_validation_curve(estimator, training_features, training_labels, title, param, parameters[param], cv)
            pylab.savefig(os.path.join(results_location, save_name))

    #set up parameters for the classifier
    if(passed_parameters is None):
        parameters = {'base_estimator__max_depth': range(1, 3), 'n_estimators' : range(5, 51, 5), 'learning_rate' : [1.0] }
    else:
        parameters = passed_parameters

    #set up tuning algorithm
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=parameters)

    #fit the classifier
    classifier.fit(training_features, training_labels)

    #get the prediction and accuracy of the test set
    test_prediction = classifier.predict(test_features)
    test_accuracy = classifier.score(test_features, test_labels)

    time_2 = time.time()

    #graph the best result
    base_classifier = tree.DecisionTreeClassifier(max_depth = classifier.best_estimator_.base_estimator_.max_depth)
    estimator = ensemble.AdaBoostClassifier(base_estimator = base_classifier, n_estimators = classifier.best_estimator_.n_estimators, learning_rate = classifier.best_estimator_.learning_rate)

    #plot the learning curve
    title = 'Learning Curves (AdaBoost - Decision Tree)\n max_depth=%i estimators=%i learning_rate=%f$' % (classifier.best_estimator_.base_estimator_.max_depth, classifier.best_estimator_.n_estimators, classifier.best_estimator_.learning_rate)
    plot_learning_curve(estimator, title, training_features, training_labels, cv=cv)
    pylab.savefig(os.path.join(results_location, 'Learning Curves - AdaBoost - Decision Tree.png'))
    
    time_3 = time.time()

    #fit the best eetimator
    estimator.fit(training_features, training_labels) 

    #plot the learning curve by number of estimators
    plot_adaclassifier(estimator, classifier.best_estimator_.n_estimators, training_features, test_features, training_labels, test_labels)
    pylab.savefig(os.path.join(results_location, 'Estimator Curves - AdaBoost - Decision Tree.png'))

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