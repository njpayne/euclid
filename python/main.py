import scipy as sp
import numpy as np 
import math
import os
import csv


data_location = "../Data" # read data from os.path.join(data_location, <filename>)
results_location = "Results" # save pdfs to os.path.join(pdf_location, <filename>)

def load_data(filename, conversion_function = None, max_records = None):
    """
    Loads the data from a csv

    Parameters
    ----------
        filename: name of csv file to be read in the data_location folder
    
    Returns
    -------
        (header, data): tuple of lists, a list of data headings and a list of data
    """
    
    header = []
    data = []

    #open the csv file
    with open(os.path.join(data_location, filename), 'rb') as csvfile:
        #read each line
        for line in csvfile.readlines():
            line_data = line.strip().split(',')
            #assume the first line is the header.  Add to the heading list
            if(len(header) == 0):
                header = line_data
            #otherwise add to the data list
            else:
                data.append(line_data)

    #just so we have a random set sort then subset
    if(max_records is not None and len(data) >= max_records):
        random.shuffle(data)
        data = data[ : max_records + 1]

    #convert the data to categories
    if(conversion_function is not None):
        header, data = conversion_function(header, data)

    return (header, data)

def main():

    test_data = load_data("test.csv")


    return

if __name__ == "__main__":
    main()