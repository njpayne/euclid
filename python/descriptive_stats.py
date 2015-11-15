import data_work
import scipy
import numpy as np
import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
import pandas as pd

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

        #load unaltered data
        headings, data = data_work.load_data(data_set + ".csv", conversion_function = None, max_records = None)

        #draw some frequency histograms
        draw_histograms(data, headings, data_set)

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

def draw_histograms(data, headings, data_set):
    """
        Chart relationships between Variables
    """
    chart_categories = ['course_grade', 'Assig_1_full_40', 'Assig_2_full_40', 'Assig_3_full_40', 'proj_1_100', 'proj_2_100', 'proj_3_100', 'final_exam_100', 'peer_feedback_100', 'birth_country', 'residence_country', 'gender', 'age', 'primary_language', 'english_fluency', 'time_zone', 'occupation', 'highest_education', 'expected_hours_spent', 'formal_class_prog_taken', 'C', 'C#', 'C++', 'Java', 'JavaScript', 'Lisp', 'Objective C', 'Perl', 'PHP', 'Python', 'Ruby', 'Shell', 'Swift', 'Visual Basic', 'Other (specify below)', 'years_programming', 'prior_omscs_classes_completed', 'besides_KBAI_how_many_classes', 'moocs_completed_outside_OMSCS', 'qtr_proj1_confidence', 'qtr_proj2_confidence', 'qtr_piazza_opinion', 'qtr_peerfeedback_opinion', 'qtr_on_piazza', 'qtr_email', 'qtr_hipchat', 'qrt_gplus', 'qtr_other_chat', 'qtr_phone', 'qtr_facebook', 'qtr_in_person', 'CS6210_Completed', 'CS8803_Completed', 'CS6250_Completed', 'CS7641_Completed', 'CS6300_Completed', 'CS6310_Completed', 'CS4495_Completed', 'CS6475_Completed', 'CS6505_Completed', 'CS6290_Completed', 'CS8803_Completed', 'CS6440_Completed', 'mid_proj2_confidence', 'mid_proj3_confidence', 'mid_piazza_opinion', 'mid_peerfeedback_opinion', 'mid_on_piazza', 'mid_email', 'mid_hipchat', 'qrt_gplus', 'mid_other_chat', 'mid_phone', 'mid_facebook', 'mid_in_person', 'final_proj3_confidence', 'hours_spent', 'lessons_watched', 'exercises_completed', 'forum_visit_frequency', 'final_on_piazza', 'final_email', 'final_hipchat', 'qrt_gplus', 'final_other_chat', 'final_phone', 'final_facebook', 'final_in_person', 'watch_out_order', 'fall_behind', 'get_ahead', 'rewatch_full_lesson', 'rewatch_partial_lesson', 'view_answer_after_1incorrect', 'repeat_exercise_until_correct', 'skip_exercise', 'correct_first_attempt', 'access_from_mobile', 'download_videos', 'piazza_answers', 'piazza_days', 'piazza_asks', 'piazza_posts', 'piazza_views', 'total_lecture_time', 'overal_lecture_views', 'lecture_1_views', 'lecture_2_views', 'lecture_3_views', 'lecture_4_views', 'lecture_5_views', 'lecture_6_views', 'lecture_7_views', 'lecture_8_views', 'lecture_9_views', 'lecture_10_views', 'lecture_11_views', 'lecture_12_views', 'lecture_13_views', 'lecture_14_views', 'lecture_15_views', 'lecture_16_views', 'lecture_17_views', 'lecture_18_views', 'lecture_19_views', 'lecture_20_views', 'lecture_21_views', 'lecture_22_views', 'lecture_23_views', 'lecture_24_views', 'lecture_25_views', 'lecture_26_views', 'lecture_1_pace', 'lecture_2_pace', 'lecture_3_pace', 'lecture_4_pace', 'lecture_5_pace', 'lecture_6_pace', 'lecture_7_pace', 'lecture_8_pace', 'lecture_9_pace', 'lecture_10_pace', 'lecture_11_pace', 'lecture_12_pace', 'lecture_13_pace', 'lecture_14_pace', 'lecture_15_pace', 'lecture_16_pace', 'lecture_17_pace', 'lecture_18_pace', 'lecture_19_pace', 'lecture_20_pace', 'lecture_21_pace', 'lecture_22_pace', 'lecture_23_pace', 'lecture_24_pace', 'lecture_25_pace', 'lecture_26_pace', 'overall_pace']
    #chart_categories = ["Age"]

    #create a folder for the dataset
    directory = os.path.dirname(os.path.join(os.getcwd(),"Results","Data Counts",data_set, ""))
    if not os.path.exists(directory):
        os.makedirs(directory)

    #convert to a pandas dataset
    pandas_data=pd.DataFrame(data = data, columns = headings)

    for chart_category in chart_categories:

        #get the slice
        index = np.argwhere(headings == chart_category)
        chart_column = data[ : , index[0][0]]


        #get counts

        plt.figure()
        plt.xlabel(chart_category)
        plt.ylabel("Count")
        plt.title("%s Count" % chart_category)

        try:
            #try converting to numbers
            chart_column = chart_column.astype(np.float)

            #create histogram
            hist, bin_edge = np.histogram(chart_column, 10)

            bin_middles = bin_edge[:-1] + np.diff(bin_edge)/2
        
            plt.hist(chart_column, 10, normed=False, histtype='bar', rwidth=0.8)

            pylab.savefig(os.path.join(os.getcwd(),"Results", "Data Counts",data_set, chart_category))

            plt.close()

        except:
            #get unique values
            unique_categories, unique_counts = np.unique(chart_column, return_counts=True)

            sns_plot = sns.countplot(x=chart_category, data=pandas_data, palette="Greens_d");
            #plt.setp(sns_plot.get_xticklabels(), rotation=45)
            sns_plot.figure.savefig(os.path.join(os.getcwd(),"Results", "Data Counts",data_set, chart_category))

            plt.close()




if __name__ == "__main__":
    main()