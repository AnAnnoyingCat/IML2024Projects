# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import math
from sklearn.metrics import mean_squared_error
# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

def fit(X, y, lam):
	"""
	This function receives training data points, then fits the ridge regression on this data
	with regularization hyperparameter lambda. The weights w of the fitted ridge regression
	are returned. 

	Parameters
	----------
	X: matrix of floats, dim = (135,13), inputs with 13 features
	y: array of floats, dim = (135,), input labels)
	lam: float. lambda parameter, used in regularization term

	Returns
	----------
	w: array of floats: dim = (13,), optimal parameters of ridge regression
	"""
	w = np.zeros((13,))

	w = CustomRidgeRegression(X, y, lam)

	assert w.shape == (13,)
	return w


def calculate_RMSE(w, X, y):
	"""This function takes test data points (X and y), and computes the empirical RMSE of 
	predicting y from X using a linear model with weights w. 

	Parameters
	----------
	w: array of floats: dim = (13,), optimal parameters of ridge regression 
	X: matrix of floats, dim = (15,13), inputs with 13 features
	y: array of floats, dim = (15,), input labels

	Returns
	----------
	RMSE: float: dim = 1, RMSE value
	"""
	RMSE = 0
	predicted_y = X.dot(w)
	diff = np.subtract(y, predicted_y)
	diff = np.square(diff)
	RMSE = np.average(diff)
	RMSE = math.sqrt(RMSE)
	assert np.isscalar(RMSE)
	return RMSE


def average_LR_RMSE(X, y, lambdas, n_folds):
	"""
	Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated, 
	and then averaged over iterations.
	
	Parameters
	---------- 
	X: matrix of floats, dim = (150, 13), inputs with 13 features
	y: array of floats, dim = (150, ), input labels
	lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
	n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV
	
	Returns
	----------
	avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
	"""
	RMSE_mat = np.zeros((n_folds, len(lambdas)))

	kf = KFold(n_splits=n_folds)

	numLam = 0
	for l in lambdas:
		numFold = 0
		for train, test in kf.split(X):
			X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
			w = fit(X_train, y_train, l)
			RMSE_mat[numFold][numLam] = calculate_RMSE(w, X_test, y_test)
			numFold += 1
		numLam += 1
	avg_RMSE = np.mean(RMSE_mat, axis=0)
	assert avg_RMSE.shape == (5,)
	return avg_RMSE

def CustomRidgeRegression(X, y, lam):
	"""
	My attempt at ridge regression. This is simply the closed form formula calculated, formula found on https://online.stat.psu.edu/stat857/node/155/
	"""
	numRows, numCols = X.shape
	I = np.identity(numCols)
	penalty = I * lam
	#calculate (XT*X + lambda*I)^-1 * XT * y; where XT is the transpose of X and I is the Identity matrix of size #cols of X.
	coeffs = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + penalty), X.T), y)

	return coeffs


# Main function. You don't have to change this
if __name__ == "__main__":
	# Data loading
	data = pd.read_csv("Task 1A\\Data\\train.csv")
	y = data["y"].to_numpy()
	data = data.drop(columns="y")
	# print a few data samples
	print(data.head())

	X = data.to_numpy()
	# The function calculating the average RMSE
	lambdas = [0.1, 1, 10, 100, 200]
	n_folds = 10
	avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
	# Save results in the required format
	np.savetxt("Task 1A\\Chris\\res.csv", avg_RMSE, fmt="%.12f")
