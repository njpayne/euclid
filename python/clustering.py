from sklearn import decomposition, cluster, feature_selection
import matplotlib.pyplot as plt
import numpy as np
import os
import pylab

data_location = "../Data" # read data from os.path.join(data_location, <filename>)
results_location = "Results" # save results text/graph to os.path.join(results_location, <filename>)

def clean_features(data, header, **kwargs):

    #extract parameters
    min_feature_variance = kwargs.get('min_feature_variance', .8 * (1 - .8))

    #remove features with variance below the threshold
    feature_selector = feature_selection.VarianceThreshold(threshold=min_feature_variance)
    reduced_data = feature_selector.fit_transform(data)
    
    #create a mask of features selected
    mask = feature_selector.get_support(indices = True)

    #select the same indexes from the header
    reduced_header = np.take(header, mask)

    return reduced_data, reduced_header

def univariate_selection(features, labels, **kwargs):

    #extract parameters
    is_regression = kwargs.get('is_regression', False)
    n_best = kwargs.get('n_best', 2)

    #select scoring function
    #For regression: f_regression
    #For classification: chi2 or f_classif
    if(is_regression):
        scoring_function = feature_selection.f_regression
    else:
        #chi2 requires non negative features
        if(features.min() < 0):
            scoring_function = feature_selection.f_classif
        else:
            scoring_function = feature_selection.chi2

    #establish the selection function
    selector = feature_selection.SelectKBest(scoring_function, k=n_best)

    #train the function
    selector.fit(features, labels.flatten())

    #get the scores
    feature_scores = selector.scores_

    #transform the data
    tranformed_data = selector.transform(features)
    
    #chart the results
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()

    X_indices = np.arange(features.shape[-1])

    plt.figure(1)
    plt.clf()
    plt.bar(X_indices - .45, scores, width=.2,
            label=r'Univariate score ($-Log(p_{value})$)', color='g')

    plt.title("Comparing feature selection")
    plt.xlabel('Feature number')
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')

    pylab.savefig(os.path.join(results_location, "Univariate Selection %d Features" % n_best))

    return tranformed_data, feature_scores

def pca_reduce(data, **kwargs):

    #extract parameters
    n_components = kwargs.get('n_components', 'mle')
    copy = kwargs.get('copy', True) 
    whiten = kwargs.get('whiten', True)

    #set up PCA function
    pca = decomposition.RandomizedPCA(n_components = n_components, copy = copy, whiten = whiten)

    #fit the data
    pca.fit(data)

    #run the reduction
    reduced_data = pca.transform(data)

    return reduced_data

def k_means_cluster(data, **kwargs):

    #extract up parameters
    n_clusters = kwargs.get('n_clusters', 10)
    n_init = kwargs.get('n_init', 10)

    #set up the clustering function
    estimator = cluster.KMeans(n_clusters = n_clusters, n_init = n_init)

    #fit the data to the training set
    estimator.fit(data)

    #transform the data
    transformed_data= estimator.transform(data)

    return transformed_data