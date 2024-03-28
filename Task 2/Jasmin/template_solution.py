# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("Task 2\\Data\\train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("Task 2\\Data\\test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables   
    """ X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1)) #not needed here, since with one-hot encoding for season we have different shapes 
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df) """

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test

   
    #One-hot encoding of season for training data, spring: 0100, summer: 0010, autumn: 1000, winter: 0001
    X_train_numerical = train_df.select_dtypes(exclude="object") #X_train with only numerical data
    X_train_nominal = train_df.select_dtypes(include="object") #X_train with only nominal data
    encoder = OneHotEncoder(sparse=False, handle_unknown="error")
    X_train_encoded = encoder.fit_transform(X_train_nominal)
    one_hot_features = pd.DataFrame(X_train_encoded)
    train_df = X_train_numerical.join(one_hot_features)
    print("TRAIN_DF W/ ONE-HOT:")
    print(train_df.head(5))

    X_array = train_df.values
     # replace missing data with the mean of the rest of the data
    imp_train = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_train.fit(X_array)
    X_array = imp_train.transform(X_array)
    
    X_train = np.delete(X_array, 1, axis=1) #drop price_CHF

    y_train = X_array[:, 1]

    # One-hot encoding of season for test-data, spring: 0100, summer: 0010, autumn: 1000, winter: 0001
    X_test_numerical = test_df.select_dtypes(exclude="object")
    X_test_nominal = test_df.select_dtypes(include="object")
    encoder_test = OneHotEncoder(sparse=False, handle_unknown="error")
    X_test_encoded = encoder_test.fit_transform(X_test_nominal)
    one_hot_features_test = pd.DataFrame(X_test_encoded)
    test_df = X_test_numerical.join(one_hot_features_test)
    print("TEST_DF W/ ONE-HOT:")
    print(test_df.head(5))
    X_test = test_df.values
    
    # replace missing data with the mean of the rest of the data
    imp_X_test = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_X_test.fit(X_test)
    X_test = imp_X_test.transform(X_test) 
    
    # End TODO

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('Task 2\\Jasmin\\results.csv', index=False)
    print("\nResults file successfully generated!")

