# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

    #Linear
    for i in range(5):
        X_transformed[:,i] = X[:, i]

    #Quadratic
    for i in range(5):
        X_transformed[:, (5 + i)] = np.power(X[:, i], 2)

    #Exponential    
    for i in range(5):
        X_transformed[:, 10 + i] = np.exp(X[:, i])
    
    #Cosine
    for i in range(5):
        X_transformed[:, 15 + i] = np.cos(X[:, i])
    
    #Constant
    X_transformed[:, 20] = np.full(700, 1)

    #print(X_transformed)
    np.savetxt("Task 1B\\Chris\\transformed.csv", X_transformed, fmt="%.12f", delimiter=',')
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
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=13)


    ITERATIONS = [5, 10, 15, 20, 100, 1000, 3500, 6500, 10000, 20000, 50000, 100000, 200000, 500000]
    LEARNING_RATES = [0.15, 0.125, 0.1, 0.09, 0.07, 0.05, 0.025]#, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001, 0.000001
    #ITERATIONS = [5, 10, 15, 20, 50, 100, 150,200, 300, 500, 700, 850, 1000]
    #LEARNING_RATES = [0.15, 0.125, 0.1]

    RMSE_mat = np.zeros((len(LEARNING_RATES), len(ITERATIONS)))
    
    currRate = 0
    for rate in LEARNING_RATES:
        currItr = 0
        for itr in ITERATIONS:
            print("attempting rate of: " + str(rate) + " at " + str(itr) + " iterations")
            w = myLinearRegression(X_train, y_train, rate, itr)
            y_pred = np.dot(X_test, w)  # find y_pred w/ help of X and w
            RMSE_mat[currRate][currItr] = mean_squared_error(y_test, y_pred, squared=False)
            print("done")
            currItr += 1
        currRate += 1
    saveManyPlots(RMSE_mat, ITERATIONS, LEARNING_RATES)

    w = myLinearRegression(X_train, y_train, 0.1, 200)
    #savePlot(ITERATIONS, cost_list)
    assert w.shape == (21,)
    return w

def myLinearRegression(X, y, learning_rate, iterations):
    """
    My attempt at linear regression. Explained and inspired by https://www.youtube.com/watch?v=sRh6w-tdtp0. 
    """
    m = y.size
    weights = np.zeros((X.shape[1]))
    #cost_list = []

    for i in range (iterations):

        y_pred = np.dot(X, weights)

        cost = (1/(2*m)) * np.sum(np.square(y_pred - y))

        d_weights = (1/m) * np.dot(X.T, (y_pred - y.T))
        uhm = (y - y_pred)
        weights = weights - learning_rate * d_weights
        #cost_list.append(cost)

        if (i % (iterations / 10) == 0):
            print("Cost is: ", cost)
    
    return weights

def savePlot(iterations, cost_list):
    rng = np.arange(0, iterations)
    plt.plot(rng, cost_list)
    plt.title(("Development of costs over the course of " + str(iterations) + " iterations."))
    plt.yscale("log")
    plt.savefig("Task 1B\\Chris\\costfunction.png")

def saveManyPlots(RMSE_mat, ITERATIONS, LEARNING_RATES):
    for r in range(len(LEARNING_RATES)):
        plt.clf()
        plt.yscale("log")
        plt.plot(ITERATIONS, RMSE_mat[r,:])
        plt.title(("Change of RMSE with learning rate " + str(LEARNING_RATES[r])))
        #plt.savefig("Task 1B\\Chris\\LR-Plots\\rate" + str(r) + ".png")
        name = "Task 1B\\Chris\\LR-Plots\\rate" + str(r) + ".png"
        plt.savefig(name)

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("Task 1B\\Data\\train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    # print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("Task 1B\\Chris\\res.csv", w, fmt="%.12f")
