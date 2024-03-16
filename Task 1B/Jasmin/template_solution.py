# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd

# Add any additional imports here (however, the task is solvable without using
# any additional imports)
# import ...
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant feature: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    # TODO: Enter your code here

    # Linear
    for i in range(5):
        X_transformed[:, i] = X[:, i]

    # Quadratic
    for i in range(5):
        X_transformed[:, (5 + i)] = np.power(X[:, i], 2)

    # Exponential
    for i in range(5):
        X_transformed[:, 10 + i] = np.exp(X[:, i])

    # Cosine
    for i in range(5):
        X_transformed[:, 15 + i] = np.cos(X[:, i])

    # Constant
    X_transformed[:, 20] = np.full(700, 1)

    # print(X_transformed)
    np.savetxt("Task 1B\\Jasmin\\transformed.csv",
               X_transformed, fmt="%.12f", delimiter=',')

    # End TODO
    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y):
    """
    This function receives training data points, transforms them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    X_transformed = transform_data(X)
    # TODO: Enter your code here

    """ lrModel = make_pipeline(StandardScaler, SGDRegressor(max_iter=10, learning_rate='constant', eta0=0.136))
    lrModel.fit(X_transformed, y)
    w = lrModel.steps[-1][1].coef_ """

    # THIS DOES NOT WORK AT ALL
    """ scaler = StandardScaler()
    scaler.fit(X_transformed)
    scaled_X_train = scaler.transform(X_transformed)
    clf = SGDRegressor(max_iter=10, learning_rate="constant", eta0=0.136)
    clf.fit(scaled_X_train, y)
    w = clf.coef_ """

    # A LITTLE BETTER, BUT STILL VERY BAD
    """ X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=13)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("RMSE: " + str(mean_squared_error(y_test, y_pred, squared=False)))
    w = clf.coef_
    plt.scatter(range(len(y_pred)), y_pred, color='blue', label='Predicted')
    plt.scatter(range(len(y_test)), y_test, color='red', label='Actual')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Comparison of Predicted and Actual Values')

    # Add legend
    plt.legend()

    # Show plot
    plt.show()
    #plt.savefig("Task 1B\\Jasmin\\plot.png") """

    # using a scaler GIVES PUBLIC SCORE OF ABOUT 280
    """ X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=13)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("RMSE w/ scaling & lin. reg: ", rmse)
    w= model.coef_ """

    # using ridge model, FINALLY HAS SIMPLE BASELINE!!!!
    """ X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=13)
    ridge = Ridge(alpha=0.02)
    ridge.fit(X_train, y_train)
    w=ridge.coef_
    y_pred = np.dot(X_test, w) #ridge.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("RMSE w/ ridge: ", rmse) """

     # using ridge and scaling, LOCAL SCORE OF ABOUT 4.5
    """X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=13)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ridge_model = Ridge(alpha=0.1)
    ridge_model.fit(X_train_scaled, y_train)
    w=ridge_model.coef_
    y_pred = np.dot(X_test, w)
    rmse=mean_squared_error(y_test, y_pred, squared=False)
    print("RMSE w/ ridge & scaling: ", rmse) """

    # using lasso regression LOCAL SCORE OF ABOUT 6.5
    """ X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=13)
    lasso_model = Lasso(alpha=1)
    lasso_model.fit(X_train, y_train)
    w = lasso_model.coef_
    y_pred = np.dot(X_test, w)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("RMSE w/ lasso: ", rmse)
 """
    # End TODO
    assert w.shape == (21,)
    return w


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("Task 1B\\Data\\train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("Task 1B\\Jasmin\\results.csv", w, fmt="%.12f")
