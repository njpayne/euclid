import scipy as sp
import numpy as np 
import math
import os
import csv

import data_work

#from sklearn import cross_validation
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler #used to convert categories to one of k 
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import zero_one_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import classifiers, clustering, regressors

data_location = "../Data" # read data from os.path.join(data_location, <filename>)
results_location = "Results" # save results text/graph to os.path.join(results_location, <filename>)


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

def run_regressors(X_train, y_train, X_test, y_test, header, cvs_writer):
    print("------------------")
    print("Running Regressors")
    print("------------------")

    print("\n\n--------------------------")
    print("Decision Trees")
    print("--------------------------")

    #create decision tree range
    decision_tree_param = {'max_depth': range(1, 200, 10)}

    #run the decision tree
    prediction, decision_tree_accuracy = regressors.run_decision_tree(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = decision_tree_param, headings = header)
    print("Decision tree accuracy = %f" % decision_tree_accuracy)

    #run the adaboost regressor
    prediction, boosting_accuracy = regressors.run_decision_tree(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = decision_tree_param)
    print("Boosting accuracy = %f" % boosting_accuracy)

    #run the random forest regressor
    prediction, rand_forest_accuracy = regressors.run_random_forest(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = decision_tree_param)
    print("Random forest accuracy = %f" % rand_forest_accuracy)

    #run the linear regressor
    prediction, linear_accuracy = regressors.run_linear_regression(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = decision_tree_param)
    print("Linear Regressor accuracy = %f" % linear_accuracy)

    #run svr
    svr_parameters = {'kernel' : ['rbf', 'poly', 'linear', 'sigmoid']}
    prediction, svr_accuracy = regressors.run_support_vector_regressor(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = svr_parameters)

    best_regressor = max(decision_tree_accuracy, boosting_accuracy, rand_forest_accuracy, linear_accuracy, svr_accuracy)

    #set up label to 
    if(header.shape[0] > 1):
        feature = ["Multi"]
    else:
        feature = [header[-1]]

    
    cvs_writer.writerow(feature + [decision_tree_accuracy, boosting_accuracy, rand_forest_accuracy, linear_accuracy, svr_accuracy])

    return best_regressor

def feature_reduction(header, data, is_classification = True):

    #remove low variance features
    data_lvar_removed, header_lvar_removed = clustering.clean_features(data, header, min_feature_variance = .8 * (1 - .8))

    #create training/test sets
    X_train, X_test, y_train, y_test = divide_for_training(data_lvar_removed)

    #select the best features using univatiate selection
    selected_features, feature_uni_scores = clustering.univariate_selection(np.vstack((X_train, X_test)), np.vstack((y_train, y_test)), n_best = 3, is_regression = not is_classification)

    #determine the order of the univariate features
    best_feature_index = np.argsort(-feature_uni_scores)

    best_results = []
  
    #reselect the features
    top_features = 12
    X_train_uni, X_test_uni = np.take(X_train, best_feature_index[ : top_features], axis = 1), np.take(X_test, best_feature_index[ : top_features], axis = 1)
    header_uni = np.take(header_lvar_removed, best_feature_index[ : top_features])

    ##try the PCA reduction
    #data_pca = clustering.pca_reduce(np.vstack((X_train, X_test)), n_components = top_features)
    #X_train_pca, X_test_pca, y_train_pca, y_test_pca = divide_for_training(np.hstack((np.vstack((y_train, y_test)), data_pca)))

    ##try the ICA reduction
    #data_ica = clustering.ica_reduce(np.vstack((X_train, X_test)), n_components = top_features)
    #X_train_ica, X_test_ica, y_train_ica, y_test_ica = divide_for_training(np.hstack((np.vstack((y_train, y_test)), data_ica)))

    #try recursive reduction
    data_recursive = clustering.recursive_reduce(np.vstack((X_train, X_test)), np.vstack((y_train, y_test)).flatten(), is_regression = not is_classification)
    X_train_recursive, X_test_recursive, y_train_recursive, y_test_recursive = divide_for_training(np.hstack((np.vstack((y_train, y_test)), data_ica)))

    if(is_classification):
        #run the classifiers and find the best result
        #best_classifier_uni = run_classifiers(X_train_uni, y_train, X_test_uni, y_test, header_uni)
        best_classifier_pca = run_classifiers(X_train_pca, y_train_pca, X_test_pca, y_test_pca, header_lvar_removed)
    else:
        #best_regressor_uni = run_regressors(X_train_uni, y_train, X_test_uni, y_test, header_uni)
        #best_regressor_pca = run_regressors(X_train_pca, y_train_pca, X_test_pca, y_test_pca, header_lvar_removed)
        #best_regressor_ica = run_regressors(X_train_ica, y_train_ica, X_test_ica, y_test_ica, header_lvar_removed)
        best_regressor_recursive = run_regressors(X_train_recursive, y_train_recursive, X_test_recursive, y_test_recursive, header_lvar_removed)
        best_results.append(max(best_regressor_ica, best_regressor_pca, best_regressor_uni))



    return

def main():
    #load the data from the csv
    header, data = data_work.load_data("basic_data.csv", conversion_function = data_work.convert_survey_data, max_records = None)

    #target categories
    select_columns = ['course_grade', 'mid_on_piazza', 'final_on_piazza', 'lecture_18_views', 'lecture_20_views', 'lecture_21_views', 'highest_education_High School', 'qtr_proj1_confidence_No Answer', 'qtr_proj1_confidence_Very unconfident', 'qtr_proj2_confidence_No Answer', 'qtr_piazza_opinion_No Answer', 'qtr_peerfeedback_opinion_No Answer', 'mid_proj2_confidence_No answer', 'mid_proj3_confidence_No answer', 'mid_piazza_opinion_No answer', 'mid_peerfeedback_opinion_No answer', 'final_proj3_confidence_No answer', 'final_proj3_confidence_Somewhat confident', 'hours_spent_No answer', 'lessons_watched_All 26', 'lessons_watched_No answer', 'exercises_completed_All of them', 'exercises_completed_No answer', 'forum_visit_frequency_No answer', 'watch_out_order_No Answer', 'fall_behind_No Answer', 'get_ahead_No Answer', 'rewatch_full_lesson_No Answer', 'rewatch_partial_lesson_No Answer', 'view_answer_after_1incorrect_No Answer', 'repeat_exercise_until_correct_No Answer', 'skip_exercise_No Answer', 'correct_first_attempt_4 - Frequently', 'correct_first_attempt_No Answer', 'access_from_mobile_Never', 'access_from_mobile_No Answer', 'download_videos_No Answer', 'lecture_11_pace_Unknown', 'lecture_12_pace_Unknown', 'lecture_13_pace_Unknown', 'lecture_14_pace_Early', 'lecture_14_pace_Unknown', 'lecture_15_pace_Unknown', 'lecture_16_pace_Unknown', 'lecture_17_pace_Unknown', 'lecture_18_pace_Unknown', 'lecture_19_pace_Unknown', 'lecture_20_pace_Unknown', 'lecture_21_pace_Unknown', 'lecture_22_pace_Unknown', 'lecture_23_pace_Unknown', 'lecture_24_pace_Unknown', 'lecture_25_pace_Early', 'lecture_25_pace_Unknown', 'lecture_26_pace_Early', 'lecture_26_pace_Unknown', 'overall_pace_Unknown']
    #select_columns = ['course_grade', 'mid_on_piazza', 'final_on_piazza']



    #select the appropriate columns
    header_subset, data_subset = data_work.select_data_columns(header, data, column_names = select_columns)

    #create csv for results
    with open(os.path.join(results_location, 'regression_results.csv'), 'wb') as output_file:

        #establish the csv writer
        writer = csv.writer(output_file, delimiter=',')

        #first run on the full set
        #assumes first column is Y
        X_train, X_test, y_train, y_test = data_work.divide_for_training(data_subset)

        #scale the data
        X_train, X_test = data_work.scale_features(X_train, X_test)

        #create headings
        writer.writerow(["Feature", "Decision Tree", "Boosting", "Random Forest", "Linear Regression", "Support Vector Machine"]) 

        #test all to start
        run_regressors(X_train, y_train, X_test, y_test, header_subset, writer)

        for i in range(0, X_train.shape[1]):
            run_regressors(X_train[:, i,np.newaxis], y_train, X_test[:, i,np.newaxis], y_test, header_subset[i + 1, np.newaxis], writer)


    return

if __name__ == "__main__":
    main()