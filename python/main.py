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
from sklearn.preprocessing import PolynomialFeatures

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

def run_regressors(X_train, y_train, X_test, y_test, header, cvs_writer, run_name = ""):

    print("\nRun Name = %s" % run_name)

    #create decision tree range
    decision_tree_param = {'max_depth': range(1, 20, 1)}

    #run the decision tree
    prediction, decision_tree_accuracy = regressors.run_decision_tree(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = decision_tree_param, headings = header, title = run_name)
    print("Decision tree accuracy = %f" % decision_tree_accuracy)

    #create adaboost range
    adaboost_parameters = {'base_estimator__max_depth': range(1, 5), 'n_estimators' : [100], 'learning_rate' : [1] }

    #run the adaboost regressor
    prediction, boosting_accuracy = regressors.run_boosting(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = adaboost_parameters)
    print("Boosting accuracy = %f" % boosting_accuracy)

    #run the random forest regressor
    prediction, rand_forest_accuracy = regressors.run_random_forest(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = decision_tree_param)
    print("Random forest accuracy = %f" % rand_forest_accuracy)

    #run the linear regressor
    prediction, linear_accuracy = regressors.run_linear_regression(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = decision_tree_param, headings = header)
    print("Linear Regressor accuracy = %f" % linear_accuracy)

    ##run svr
    #svr_parameters = {'kernel' : ['rbf', 'poly', 'linear', 'sigmoid'], 'degree' : [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    #prediction, svr_accuracy = regressors.run_support_vector_regressor(X_train, y_train.flatten(), X_test, y_test.flatten(), passed_parameters = svr_parameters)
    #print("SVM accuracy = %f" % svr_accuracy)

    svr_accuracy = 0

    best_regressor = max(decision_tree_accuracy, boosting_accuracy, rand_forest_accuracy, linear_accuracy, svr_accuracy)
    
    cvs_writer.writerow([run_name] + [decision_tree_accuracy, boosting_accuracy, rand_forest_accuracy, linear_accuracy, svr_accuracy] + [""] + header)

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


    #create a dictionary of feature setups
    #select the appropriate columns
    feature_dict = {
        #"Full" : header.tolist(),
        #"Lecture Views" : ['overal_lecture_views', 'total_lecture_time'],
        #"Piazza Use" : ['mid_on_piazza', 'final_on_piazza', 'piazza_posts', 'piazza_days', 'piazza_views'],
        #"Lecture Pace" : ['lecture_1_pace_Late', 'lecture_1_pace_On-time', 'lecture_1_pace_Unknown', 'lecture_2_pace_Late', 'lecture_2_pace_On-time', 'lecture_2_pace_Unknown', 'lecture_3_pace_Late', 'lecture_3_pace_On-time', 'lecture_3_pace_Unknown', 'lecture_4_pace_Early', 'lecture_4_pace_Late', 'lecture_4_pace_On-time', 'lecture_4_pace_Unknown', 'lecture_5_pace_Early', 'lecture_5_pace_Late', 'lecture_5_pace_On-time', 'lecture_5_pace_Unknown', 'lecture_6_pace_Early', 'lecture_6_pace_Late', 'lecture_6_pace_On-time', 'lecture_6_pace_Unknown', 'lecture_7_pace_Early', 'lecture_7_pace_Late', 'lecture_7_pace_On-time', 'lecture_7_pace_Unknown', 'lecture_8_pace_Early', 'lecture_8_pace_Late', 'lecture_8_pace_On-time', 'lecture_8_pace_Unknown', 'lecture_9_pace_Early', 'lecture_9_pace_Late', 'lecture_9_pace_On-time', 'lecture_9_pace_Unknown', 'lecture_10_pace_Early', 'lecture_10_pace_Late', 'lecture_10_pace_On-time', 'lecture_10_pace_Unknown', 'lecture_11_pace_Early', 'lecture_11_pace_Late', 'lecture_11_pace_On-time', 'lecture_11_pace_Unknown', 'lecture_12_pace_Early', 'lecture_12_pace_Late', 'lecture_12_pace_On-time', 'lecture_12_pace_Unknown', 'lecture_13_pace_Early', 'lecture_13_pace_Late', 'lecture_13_pace_On-time', 'lecture_13_pace_Unknown', 'lecture_14_pace_Early', 'lecture_14_pace_Late', 'lecture_14_pace_On-time', 'lecture_14_pace_Unknown', 'lecture_15_pace_Early', 'lecture_15_pace_Late', 'lecture_15_pace_On-time', 'lecture_15_pace_Unknown', 'lecture_16_pace_Early', 'lecture_16_pace_Late', 'lecture_16_pace_On-time', 'lecture_16_pace_Unknown', 'lecture_17_pace_Early', 'lecture_17_pace_Late', 'lecture_17_pace_On-time', 'lecture_17_pace_Unknown', 'lecture_18_pace_Early', 'lecture_18_pace_Late', 'lecture_18_pace_On-time', 'lecture_18_pace_Unknown', 'lecture_19_pace_Early', 'lecture_19_pace_Late', 'lecture_19_pace_On-time', 'lecture_19_pace_Unknown', 'lecture_20_pace_Early', 'lecture_20_pace_Late', 'lecture_20_pace_On-time', 'lecture_20_pace_Unknown', 'lecture_21_pace_Early', 'lecture_21_pace_Late', 'lecture_21_pace_On-time', 'lecture_21_pace_Unknown', 'lecture_22_pace_Early', 'lecture_22_pace_Late', 'lecture_22_pace_On-time', 'lecture_22_pace_Unknown', 'lecture_23_pace_Early', 'lecture_23_pace_Late', 'lecture_23_pace_On-time', 'lecture_23_pace_Unknown', 'lecture_24_pace_Early', 'lecture_24_pace_Late', 'lecture_24_pace_On-time', 'lecture_24_pace_Unknown', 'lecture_25_pace_Early', 'lecture_25_pace_Late', 'lecture_25_pace_On-time', 'lecture_25_pace_Unknown', 'lecture_26_pace_Early', 'lecture_26_pace_Late', 'lecture_26_pace_On-time', 'lecture_26_pace_Unknown', 'overall_pace_Early', 'overall_pace_Late', 'overall_pace_On-time', 'overall_pace_Unknown'],
        #"Classmate Contact" : ['qtr_on_piazza', 'qtr_email', 'qtr_hipchat', 'qrt_gplus', 'qtr_other_chat', 'qtr_phone', 'qtr_facebook', 'qtr_in_person', 'mid_on_piazza', 'mid_email', 'mid_hipchat', 'qrt_gplus', 'mid_other_chat', 'mid_phone', 'mid_facebook', 'mid_in_person', 'final_on_piazza', 'final_email', 'final_hipchat', 'qrt_gplus', 'final_other_chat', 'final_phone', 'final_facebook', 'final_in_person'],
        #"Lecture Amount" : ['total_lecture_time', 'overal_lecture_views', 'lecture_1_views', 'lecture_2_views', 'lecture_3_views', 'lecture_4_views', 'lecture_5_views', 'lecture_6_views', 'lecture_7_views', 'lecture_8_views', 'lecture_9_views', 'lecture_10_views', 'lecture_11_views', 'lecture_12_views', 'lecture_13_views', 'lecture_14_views', 'lecture_15_views', 'lecture_16_views', 'lecture_17_views', 'lecture_18_views', 'lecture_19_views', 'lecture_20_views', 'lecture_21_views', 'lecture_22_views', 'lecture_23_views', 'lecture_24_views', 'lecture_25_views', 'lecture_26_views'],
        #"Prior Experience" : ['formal_class_prog_taken', 'C', 'C#', 'C++', 'Java', 'JavaScript', 'Lisp', 'Objective C', 'Perl', 'PHP', 'Python', 'Ruby', 'Shell', 'Swift', 'Visual Basic', 'Other (specify below)', 'years_programming', 'prior_omscs_classes_completed', 'occupation', 'highest_education', 'besides_KBAI_how_many_classes', 'moocs_completed_outside_OMSCS'],
        "Self Assesment" : ['qtr_proj1_confidence_neither confident nor unconfident', 'qtr_proj1_confidence_no answer', 'qtr_proj1_confidence_somewhat confident', 'qtr_proj1_confidence_somewhat unconfident', 'qtr_proj1_confidence_very confident', 'qtr_proj1_confidence_very unconfident', 'qtr_proj2_confidence_neither confident nor unconfident', 'qtr_proj2_confidence_no answer', 'qtr_proj2_confidence_somewhat confident', 'qtr_proj2_confidence_somewhat unconfident', 'qtr_proj2_confidence_very confident', 'qtr_proj2_confidence_very unconfident', 'mid_proj2_confidence_neither confident nor unconfident', 'mid_proj2_confidence_no answer', 'mid_proj2_confidence_somewhat confident', 'mid_proj2_confidence_somewhat unconfident', 'mid_proj2_confidence_very confident', 'mid_proj2_confidence_very unconfident', 'mid_proj3_confidence_neither confident nor unconfident', 'mid_proj3_confidence_no answer', 'mid_proj3_confidence_somewhat confident', 'mid_proj3_confidence_somewhat unconfident', 'mid_proj3_confidence_very confident', 'mid_proj3_confidence_very unconfident', 'final_proj3_confidence_neither confident nor unconfident', 'final_proj3_confidence_no answer', 'final_proj3_confidence_somewhat confident', 'final_proj3_confidence_somewhat unconfident', 'final_proj3_confidence_very confident', 'final_proj3_confidence_very unconfident']
    }

    #list the data sources
    data_sources = [
        #"basic_data", 
        #"basic_data_only_finishers", 
        #"basic_data_clean_lecture", 
        "basic_data_piazza",
        #"basic_data_piazza_only_finishers"
        ]


    #create csv for results
    with open(os.path.join(results_location, 'regression_results.csv'), 'wb') as output_file:
            
        #establish the csv writer
        writer = csv.writer(output_file, delimiter=',')

        for data_source in data_sources:

            print("\n\n------------------")
            print("Data Set - %s" % data_source)
            print("------------------")

            #this section determines R^2 scores of the regressors
            writer.writerow(["R^2 Scores"])

            writer.writerow(["Dataset - %s" % data_source] )

            #load the data from the csv
            header, data = data_work.load_data(data_source + ".csv", conversion_function = data_work.convert_survey_data, max_records = None)

            #create headings
            writer.writerow(["Feature", "Decision Tree", "Boosting", "Random Forest", "Linear Regression", "Support Vector Machine", "", "Feature Details"]) 

            #loop through all the feature set combos
            for feature_set_name, select_columns in feature_dict.iteritems():

                print("\n\n------------------")
                print("Feature Set - %s" % feature_set_name)
                print("------------------")

                #get the data subset
                header_subset, data_subset = data_work.select_data_columns(header, data, column_names = ['course_grade'] + select_columns)

                #first run on the full set
                #assumes first column is Y
                X_train, X_test, y_train, y_test = data_work.divide_for_training(data_subset)

                #remove the label header
                header_subset = header_subset[1 : ]

                #scale the data
                X_train, X_test = data_work.scale_features(X_train, X_test)

                #test all to start
                run_regressors(X_train, y_train, X_test, y_test, header_subset, writer, data_source + "-" + feature_set_name + "Linear")

                #for degree in [2, 3, 4]:
                for degree in [2]:
                    #convert to polynomials
                    poly = PolynomialFeatures(degree=degree)
                    X_train_poly = poly.fit_transform(X_train)
                    X_test_poly = poly.fit_transform(X_test)

                    #test all in poly
                    run_regressors(X_train_poly , y_train, X_test_poly, y_test, header_subset, writer, data_source + "-" + feature_set_name + "Poly %i" % degree)

                ##test individually
                #for i in range(0, X_train.shape[1]):
                #    run_regressors(X_train[:, i,np.newaxis], y_train, X_test[:, i,np.newaxis], y_test, header_subset[i + 1, np.newaxis], writer)


    return

if __name__ == "__main__":
    main()