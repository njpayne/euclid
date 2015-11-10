import data_work
import scipy
import numpy as np
import csv
import os

data_location = "../Data" # read data from os.path.join(data_location, <filename>)
results_location = "Results" # save results text/graph to os.path.join(results_location, <filename>)

def main():

    #load the data
    headings, data = data_work.load_data("basic_data.csv", conversion_function = data_work.convert_survey_data, max_records = None)

    #get the statistics
    generate_statistics(data, headings)


    return

def generate_statistics(data, headings):
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
    with open(os.path.join(results_location, 'descriptive_stats.csv'), 'wb') as output_file:
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
    return

if __name__ == "__main__":
    main()