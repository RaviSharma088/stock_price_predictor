import numpy as np
import pandas as pd
import stock_data as sd
import visualize as vs
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Apple_Preprocessed.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)
X_test=sc_X.fit_transform(X_test)

print("x_train", X_train.shape)
print("y_train", y_train.shape)
print("x_test", X_test.shape)
print("y_test", y_test.shape)

# Fitting DecisionTree to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(y_test)





#Graph plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Add labels
plt.ylabel("Price USD")
plt.xlabel("Trading Days")

# Plot actual and predicted close values

plt.plot(y_pred, '#00FF00', label='Predicted Close')
plt.plot(y_test, '#0000FF', label='Actual Close')

# Set title
ax.set_title("Trading vs Prediction")
ax.legend(loc='upper left')


plt.show()