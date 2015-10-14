# Initial data aggregation & summary analysis
# File created October 11, 2015, Nathaniel Payne
# Educational Data Mining Project; Georgia Institute of Technology

set.seed(120581724) # Set a random seed so that the random sample is regeneratable

# --------------------------------------------------------------------------------------------
# Install & import all relevant libraries

# install.packages("dummies")
# install.packages("data.table")
# install.packages("sqldf") # Load in the SQLdf package
# install.packages("plyr") # Install the plyr package
# install.packages("rattle") # Install the rattle package
# install.packages("nnet") # Install the neural net package
# install.packages("e1071") # Install the package in R used for SVM's
# install.packages("ipred") # Install the package used for K-NN in R
# install.packages("class") # Installing the class package
# install.packages("adabag") # Create boosted decision trees
# install.packages("aod")

library(dummies)
library(data.table) # Used for faster table manipulation
library(sqldf) # Run SQL in R
library(plyr) # Load the plyr package
library(rpart) # Turn on the rpart library
library(rattle) # Used for making fancy classification & regression trees
library(nnet) # Load the nnet library for use with neural net analysis
library(e1071) # Load the package used for SVM's
library(ipred) # Load the library used for k-NN simulation
library(class) # Load the class library for use with kNN
library(adabag) # Used for boosting decision trees
library(aod)
library(ggplot2)

# --------------------------------------------------------------------------------------------
# Load in the initial test gradebook data

# Set the main working directory to load in the data (switch to make this local)
setwd("C:/Users/nathanielpayne.CPATHAD/Dropbox/Data") # Set the main working directory

# Dataset 1 = Gradebook
gradebook <- read.csv("gradebook.csv", sep = ",", header = TRUE)

# Review the imported data
head(gradebook) # Review the gradebook names
colnames(gradebook) # Review the gradebook

# Need to change the gradebook headings
colnames(gradebook) <- c("student_id", "ass_1_40", "assi_2_40", "proj1_100", "assi_3_40", "assi_4_40", 
                         "proj2_100", "assi_5_40", "assi_6_40", "proj3_100", "final_exam_100", 
                         "peer_feedback_15", "final_course_grade")

str(gradebook) # Review the structure of the data
head(gradebook) # Review the gradebook

# Review a histogram of grades (Note the skewness)
jpeg('final_course_grade_histogram.jpg')
hist(gradebook$final_course_grade, 
     breaks = c(0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100), 
     freq = TRUE, 
     main = "Histogram of Final Course Grades",
     xlab = "Final Course Grade",
     ylab = "Frequency",
     col = "Red")
dev.off() # Turn off

# --------------------------------------------------------------------------------------------
# Load in the full gradebook data

# --------------------------------------------------------------------------------------------
# Complete summary analysis on the data & generate final initial graphs

# --------------------------------------------------------------------------------------------
# Determine the final dataset to be worked on for modelling

# --------------------------------------------------------------------------------------------
# --------------------------------  Move section to a new program ----------------------------
# --------------------------------------------------------------------------------------------
# Split the dataset into 20 / 80 test / train datasets

# --------------------------------------------------------------------------------------------
# Fit the data using all subsets regression, ridge regression, the LASSO, random forest, neural network
# Basic regression tree models (use all subsets and the LASSO for variable selection)

# --------------------------------------------------------------------------------------------
# Analyze the resulting output & test significance

