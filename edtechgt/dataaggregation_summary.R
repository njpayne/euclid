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
# Load in relevant data

setwd("c:/Users/nathanielpayne/Documents/DeVry") # Set the main working directory