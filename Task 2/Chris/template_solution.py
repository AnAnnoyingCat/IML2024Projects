# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sb
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

"""
The plan:
Model non-numeric data as one hot columns. Generate different files using different methods of filling in the missing data.
Perform cross validation of kernels on all of them, select the best model and train that.
"""

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
    
    # Encode seasons of training data
    season_column = train_df[['season']]
    seasons = ['spring', 'summer', 'autumn', 'winter']
    enc = preprocessing.OneHotEncoder(categories=[seasons], sparse=False)
    oneHotSeasons = enc.fit_transform(season_column)
    onehot_seasons_df = pd.DataFrame(oneHotSeasons,  columns=enc.get_feature_names_out(['season']))
    X_train = pd.concat([onehot_seasons_df,train_df.drop(['price_CHF', 'season'],axis=1)], axis=1)

    # Extract traning y
    y_train = train_df[['price_CHF']]

    # Encode seasons of testing data
    season_column = test_df[['season']]
    oneHotSeasons = enc.fit_transform(season_column)
    onehot_seasons_df = pd.DataFrame(oneHotSeasons,  columns=enc.get_feature_names_out(['season']))
    X_test = pd.concat([onehot_seasons_df,test_df.drop(['season'],axis=1)], axis=1)
    print(X_train)
    print(y_train)
    print(X_test)
    
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
    generate_missing_values_files()
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('Task 2/Chris/results.csv', index=False)
    print("\nResults file successfully generated!")
    

"""
def generate_missing_values_files():
    # Import data and onehot-encode seasons
    train_df = pd.read_csv("Task 2\\Data\\train.csv")
    train_df_price_values = train_df.drop(['season'], axis=1)

    # Split off and handle the seasons
    season_column = train_df[['season']]
    seasons = ['spring', 'summer', 'autumn', 'winter']
    enc = preprocessing.OneHotEncoder(categories=[seasons], sparse=False)
    oneHotSeasons = enc.fit_transform(season_column)
    onehot_seasons_df = pd.DataFrame(oneHotSeasons,  columns=enc.get_feature_names_out(['season']))
    print(onehot_seasons_df)

    # Univariante
    ##Avg_colwise
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    avg_data = imp.fit_transform(train_df_price_values)
    avg_data_df1 = pd.DataFrame(avg_data, columns=train_df_price_values.columns)
    full_data = pd.concat([onehot_seasons_df,avg_data_df1], axis=1)
    full_data.to_csv("Task 2/Data/filled_in_data_avg_colwise.csv", index=False)

    ##Median_colwise
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    avg_data = imp.fit_transform(train_df_price_values)
    avg_data_df2 = pd.DataFrame(avg_data, columns=train_df_price_values.columns)
    full_data = pd.concat([onehot_seasons_df,avg_data_df2], axis=1)
    full_data.to_csv("Task 2/Data/filled_in_data_median_colwise.csv", index=False)

    ##Avg_rowwise
    # Hypothesis: prices between countries are correlated, so this might work.
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    avg_data = imp.fit_transform(train_df_price_values.transpose())
    avg_data_df3 = pd.DataFrame(avg_data.transpose(), columns=train_df_price_values.columns)
    full_data = pd.concat([onehot_seasons_df,avg_data_df3], axis=1)
    full_data.to_csv("Task 2/Data/filled_in_data_avg_rowwise.csv", index=False)

    ##Median_rowwise
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    avg_data = imp.fit_transform(train_df_price_values.transpose())
    avg_data_df4 = pd.DataFrame(avg_data.transpose(), columns=train_df_price_values.columns)
    full_data = pd.concat([onehot_seasons_df,avg_data_df4], axis=1)
    full_data.to_csv("Task 2/Data/filled_in_data_median_rowwise.csv", index=False)

    # Multivariante
    ## Iterative Imputer

    imp = IterativeImputer(max_iter=126, random_state=13)
    avg_data = imp.fit_transform(train_df_price_values)
    avg_data_df = pd.DataFrame(avg_data, columns=train_df_price_values.columns)
    full_data = pd.concat([onehot_seasons_df, avg_data_df], axis=1)
    full_data.to_csv("Task 2/Data/filled_in_data_iterimp.csv", index=False)

    # KNN
    
    imp = KNNImputer(n_neighbors=4,weights='uniform')
    avg_data = imp.fit_transform(train_df_price_values)
    avg_data_df = pd.DataFrame(avg_data, columns=train_df_price_values.columns)
    full_data = pd.concat([onehot_seasons_df, avg_data_df], axis=1)
    full_data.to_csv("Task 2/Data/filled_in_data_knn.csv", index=False)
"""