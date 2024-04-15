# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, ConstantKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import seaborn as sns
from joblib import Parallel, delayed

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
    train_df = pd.read_csv("Task 2\\Data\\filled_in_data_knn.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("Task 2\\Data\\test_filled_in_data_knn.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))
    
    X_train = train_df.drop('')

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

# defineXY as implemented by Jasmin
def defineXY(file_train, file_test):
    df_train = pd.read_csv("{0}".format(file_train))
    X_train = df_train.drop(['price_CHF'], axis=1)
    y = df_train['price_CHF']
    df_test = pd.read_csv("{0}".format(file_test))
    return X_train.values, y.values, df_test.values


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
    #using the knn data as it seems like the best fit
    #trying different things for different kernels and just seeing what works.
    #i implemented this in a parallel manner so that it doesn't take way too long

    #Things to try:
    constants_to_try = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    initial_lengths_to_try = [0.0001,0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 4, 8, 16]
    alphas_to_try = [0.1, 0.2, 0.4, 0.8, 1, 1.2, 1.4, 1.6, 2, 3, 4]

    #Trying them in parallel to speed things up, since the different argument combinations ar emutually exclusive.
    results = Parallel(n_jobs=-1)(
        delayed(try_combinations)(X_train, y_train, X_test, constant, length, alpha)
        for constant in constants_to_try
        for length in initial_lengths_to_try
        for alpha in alphas_to_try
    )
    #plot the results to a seaborne pairplot
    plot_results(results)

    #print the best indices
    opt_index = np.argmax(scores)

    # Extract the optimal result
    optres = results[opt_index]

    # Extract optimal values
    optimal_constant = optres['constant']
    optimal_length = optres['length']
    optimal_alpha = optres['alpha']
    optimal_score = optres['score']
    print(f"Optimal result with constant {optimal_constant}, length {optimal_length}, and alpha {optimal_alpha}. R2 score: {optimal_score}")

    # y_pred = gp.predict(X_test)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

def try_combination_of_params_on_rbf(X_train_train, y_train_train, X_train_test, y_train_test, constant, init_length, alpha):
    #init the kernel using current params
    kernel = ConstantKernel(constant_value=c) * RBF(length_scale=initlength, length_scale_bounds=(1e-5, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)
    gp.fit(X_train, y_train)
    y_train_predict = gp.predict(X_train_test)
    localScore = r2_score(y_train_test, y_train_predict)

    print(f"Done with combo: constant={constant}, length={length}, alpha={alpha}")

    return {'constant': constant, 'init_length': init_length, 'alpha': alpha, 'score': local_score}

def plot_results(res):
    #using seaborne pairplot as shown in the lecture
    df = pd.DataFrame(res)
    sns.pairplot(df, vars=['constant', 'length', 'alpha', 'score'])
    pairplot.savefig('pairplot.png')
    

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = defineXY("Task 2\\Data\\filled_in_data_knn.csv", "Task 2\\Data\\test_filled_in_data_knn.csv")
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