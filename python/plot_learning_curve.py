##
##code borrowed from:
##http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html 
##and http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn.metrics import zero_one_loss
import math

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_learning_curve_iter(estimator, title, cv = 10):

    test_scores = []
    test_std =[]
    iteration_count = []

    for i in range(len(estimator.grid_scores_)):
        #get the cross validation results from the estimator
        iteration_count.append(estimator.grid_scores_[i][0]["n_iter"])
        test_scores.append(estimator.grid_scores_[i].mean_validation_score)
        test_std.append(np.std(estimator.grid_scores_[i].cv_validation_scores))

    #convert arrays to numpy
    test_scores = np.array(test_scores)
    test_std = np.array(test_std)
    iteration_count = np.array(iteration_count)

    plt.figure()
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.grid()
    #plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="g")
    plt.fill_between(iteration_count, test_scores - test_std ,
                     test_scores + test_std , alpha=0.2, color="g")
    plt.plot(iteration_count, test_scores, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")

    return plt

def plot_validation_curve(estimator, X, y, title, param_name, param_range, cv = 10):
    train_scores, test_scores = validation_curve(
    estimator, X, y, param_name=param_name, param_range=param_range,
    cv=cv, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    #plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    #plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_adaclassifier(classifier, n_estimators, X_train, X_test, y_train, y_test):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-',
    #        label='Decision Stump Error')
    #ax.plot([1, n_estimators], [dt_err] * 2, 'k--',
    #        label='Decision Tree Error')

    ada_err_test = np.zeros((n_estimators,))
    for i, y_pred in enumerate(classifier.staged_predict(X_test)):
        ada_err_test[i] = zero_one_loss(y_pred, y_test)

    ada_err_train = np.zeros((n_estimators,))
    for i, y_pred in enumerate(classifier.staged_predict(X_train)):
        ada_err_train[i] = zero_one_loss(y_pred, y_train)

    ax.plot(np.arange(n_estimators) + 1, ada_err_test,
            label='AdaBoost Test Error',
            color='red')
    ax.plot(np.arange(n_estimators) + 1, ada_err_train,
            label='AdaBoost Train Error',
            color='blue')

    ax.set_ylim((0.0, 1.0))
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('error rate')

    leg = ax.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.7)


    return fig