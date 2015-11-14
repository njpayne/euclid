import data_work
import scipy
import numpy as np
import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pylab

data_location = "../Data" # read data from os.path.join(data_location, <filename>)
results_location = "Results" # save results text/graph to os.path.join(results_location, <filename>)

def main():

    #run the two datasets
    data_sets = ["basic_data", "basic_data_only_finishers", "basic_data_clean_lecture", "basic_data_piazza",  "basic_data_piazza_only_finishers"]

    for data_set in data_sets:

        #load the data
        headings, data = data_work.load_data(data_set + ".csv", conversion_function = data_work.convert_survey_data, max_records = None)

        #generate the stats
        generate_stats(data, headings, data_set)

        #draw charts for the relationships between variables
        draw_charts(data, headings, data_set)

    return

def generate_stats(data, headings, data_set):
    """
        Runs the basic descriptive stats for the dataset
    """

    #get the basic descriptive statistics
    basic_descriptive_stats = scipy.stats.describe(data)

    #unbundle the results
    kurtosis = basic_descriptive_stats.kurtosis
    mean = basic_descriptive_stats.mean
    skewness = basic_descriptive_stats.skewness
    variance = basic_descriptive_stats.variance
    covariance = np.cov(data, rowvar = 0)
    pearson_correlation = np.corrcoef(data, rowvar = 0)

    #create output csv
    with open(os.path.join(results_location, 'descriptive_stats' + data_set +  '.csv'), 'wb') as output_file:
        #establish the csv writer
        writer = csv.writer(output_file, delimiter=',')

        #write the data
        writer.writerow(np.append(np.array(["Statistics"]), headings))
        writer.writerow(np.append(np.array(["Kurtosis"]), kurtosis))
        writer.writerow(np.append(np.array(["Mean"]), mean))
        writer.writerow(np.append(np.array(["Skewness"]), skewness))
        writer.writerow(np.append(np.array(["Variance"]), variance))

        #write the covariance matrix
        writer.writerow([""])
        writer.writerow(["Covariance"])
        writer.writerow(np.append(np.array([""]), headings))
        writer.writerows(np.hstack((np.expand_dims(headings, axis = 1), covariance)))
        
        #write the pearson correlation coefficient
        writer.writerow([""])
        writer.writerow(["Pearson Correlation"])
        writer.writerow(np.append(np.array([""]), headings))
        writer.writerows(np.hstack((np.expand_dims(headings, axis = 1), pearson_correlation)))

def draw_charts(data, headings, data_set):
    """
        Chart relationships between Variables
    """

    #create a folder for the dataset
    directory = os.path.dirname(os.path.join(os.getcwd(),"Results",data_set, ""))
    if not os.path.exists(directory):
        os.makedirs(directory)

    #just plot vs course grades
    for i in range(data.shape[1]):
        for j in np.argwhere(headings == "course_grade")[0].tolist():

            #no need to test against itself
            #or earlier combos
            if(j != i):
                x_values = data[ : , i]
                y_values = data[ : , j]

                x_label = headings[i]
                y_label = headings[j]

                #make a scatterplot
                plt.figure()
                plt.scatter(x_values, y_values)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                #plt.legend(loc='upper right')
                plt.title(x_label + " Vs. " + y_label)

                #save the graph to file
                save_name =  x_label + "_" + y_label
                save_name = save_name.replace("<", "lt")
                save_name = save_name.replace(">", "gt")
                save_name = save_name.replace("\\", "")
                save_name = save_name.replace("/", "")
                save_name = save_name.replace(".", "")
                save_name = save_name.replace("I have not used Piazza significantly", "")

                pylab.savefig(os.path.join(os.getcwd(),"Results",data_set,save_name))

                plt.close()
    return

if __name__ == "__main__":
    main()