# Import necessary libraries
import pandas as pd
import itertools
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read the file "Advertising.csv"
df = pd.read_csv("Advertising.csv")
# Take a quick look at the data to list all the predictors
df.head()
### edTest(test_mse) ###

# Initialize a list to store the MSE values
mse_list = []

# Create a list of lists of all unique predictor combinations
# For example, if you have 2 predictors,  A and B, you would
# end up with [['A'],['B'],['A','B']]
predictors = list(df.columns)
predictors.remove('Sales')
print(predictors)

# Generate all unique combinations of the predictors
cols = []
for r in range(1, len(predictors) + 1):
    combinations = itertools.combinations(predictors, r)
    cols.extend(combinations)

# Convert tuples to lists
cols = [list(comb) for comb in cols]

# Loop over all the predictor combinations
for i in cols:
    # Set each of the predictors from the previous list as x
    x = df[i]

    # Set the "Sales" column as the reponse variable
    y = df[['Sales']]

    # Split the data into train-test sets with 80% training data and 20% testing data.
    # Set random_state as 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

    # Initialize a Linear Regression model
    lreg = LinearRegression()

    # Fit the linear model on the train data
    lreg.fit(x_train, y_train)

    # Predict the response variable for the test set using the trained model
    y_pred = lreg.predict(x_test)

    # Compute the MSE for the test data
    MSE = mean_squared_error(y_test, y_pred)

    # Append the computed MSE to the initialized list
    mse_list.append(MSE)
# Helper code to display the MSE for each predictor combination
t = PrettyTable(['Predictors', 'MSE'])

for i in range(len(mse_list)):
    t.add_row([cols[i],round(mse_list[i],3)])

print(t)
