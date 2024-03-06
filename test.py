import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, PredictionErrorDisplay

#training data
trainingData = pd.read_csv('train.csv', sep=',')
X = trainingData.iloc[:,2:12]
y = trainingData['y']

#prediction data
predData = pd.read_csv('test.csv', sep=',')
X_pred = predData.iloc[:,1:11]
id = predData.iloc[:, 0]

#splitting data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.33, random_state= 101)

# Instantiating LinearRegression() Model
lr = linear_model.LinearRegression()

# Training/Fitting the Model
lr.fit(X_train, y_train)

# Making Predictions
lr.predict(X_test)
pred = lr.predict(X_test)

# The coefficients
print("Coefficients: \n", lr.coef_)
# Evaluating Model's Performance
print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print('Mean Squared Error:', mean_squared_error(y_test, pred))
print('Mean Root Squared Error:', np.sqrt(mean_squared_error(y_test, pred)))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, pred))

#predict the test data
pred = lr.predict(X_pred)

results = pd.DataFrame({
	'Id': id,
	'y': pred
})
print("Saving Results to \"prediction.csv\"...")

results.to_csv('prediction.csv', sep=',', index=False)