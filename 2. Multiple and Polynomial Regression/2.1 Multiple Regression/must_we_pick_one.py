# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import fit_and_plot_linear, fit_and_plot_multi

# Read the file "Advertising.csv"
df = pd.read_csv("Advertising.csv")

# Take a quick look at the dataframe
df.head()
# Define an empty Pandas dataframe to store the R-squared value associated with each
# predictor for both the train and test split
df_results = pd.DataFrame(columns=['Predictor', 'R2 Train', 'R2 Test'])
# For each predictor in the dataframe, call the function "fit_and_plot_linear()"
# from the helper file with the predictor as a parameter to the function

# This function will split the data into train and test split, fit a linear model
# on the train data and compute the R-squared value on both the train and test data
results = []
for predictor in df:
	r2_train, r2_test = fit_and_plot_linear(df[[predictor]])
	results.append((predictor, r2_train, r2_test))
### edTest(test_chow1) ###
# Submit an answer choice as a string below
# (Eg. if you choose option C, put 'C')
answer1 = 'A'
# Call the function "fit_and_plot_multi()" from the helper to fit a multilinear model
# on the train data and compute the R-squared value on both the train and test data

r2_train, r2_test = fit_and_plot_multi()
### edTest(test_dataframe) ###

# Store the R-squared values for all models
# in the dataframe intialized above
df_results = pd.DataFrame(results, columns=['Predictor', 'R2 Train', 'R2 Test'])

# Take a quick look at the dataframe
df_results.head()
### edTest(test_chow2) ###
# Submit an answer choice as a string below
# (Eg. if you choose option C, put 'C')
answer2 = 'B'
