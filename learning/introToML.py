# Libraries import

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Opening data

melbourne_file_path = './learning/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Printing general stats
print(melbourne_data.describe())

# Printing column names
print(melbourne_data.columns)

# dropping rows (axis=0) drops missing values
melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price

print("Price:")
print(y)

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]

print("Features:")
print(X.describe())

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are:")
print(melbourne_model.predict(X.head()))

predicted_home_prices = melbourne_model.predict(X)
error = mean_absolute_error(y, predicted_home_prices)
print("The mean absolute error (MAE) is:")
print(error)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))