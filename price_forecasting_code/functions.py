import pandas as pd
import numpy as np
np.random.seed(seed = 1)
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
from config import ameritrade_credentials
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import itertools
import statsmodels.api as sm
import warnings
import tensorflow as tf
tf.random.set_seed(2)
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.callbacks import History
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')



def get_data_from_ameritrade(ticker_list, payload):
    """
    Takes in list of tickers and payload and returns complete dataframe with all 
    tickers and prices.
    
    ticker_list -- array of strings containing n number of tickers
    payload -- dictionary containing parameters for API call
    
    """
    df_list = []
    fail_cache = []
    payload = payload
    for ticker in ticker_list:
        # Define endpoint
        endpoint = r'https://api.tdameritrade.com/v1/marketdata/{}/pricehistory'.format(ticker)
        
        try:
            # Make Request
            content = requests.get(url = endpoint, params = payload)

            # Convert to dataframe
            data = content.json()
            data = pd.DataFrame(data = data['candles'])
            data['ticker'] = ticker

            # Append to list of dataframes
            df_list.append(data)
            
        except:
            # Append problematic ticker to list of failed tickers
            fail_cache.append(ticker)
            continue
    
    return df_list, fail_cache



def clean_dates(df_list):
    """
    Takes in a list of dfs and cleans the dates for each df.
    
    df_list -- list of dataframes
    """
    for df in df_list:
        # Convert from unix to year month day format
        df['datetime'] = pd.to_datetime(df['datetime'], unit = 'ms')

        # Set datetime as index
        df.set_index(df['datetime'], inplace = True)

        # Drop original datetime column
        df.drop(columns = 'datetime', inplace = True)
        
    return df_list



def create_intra_clf_target(df_list):
    """
    Takes in list of dfs and creates target variable for each depending
    on if the closing price for each day exceeds the opening price. If the
    closing price exceeds the opening price, target = 1.
    
    df_list -- list of dfs
    """
    for df in df_list:
        df['intra_clf_target'] = 0
        for i in range(len(df)):
            
            if df['open'][i] < df['close'][i]:
                df['intra_clf_target'][i] = 1
            
            elif df['open'][i] > df['close'][i]:
                df['intra_clf_target'][i] = 0
    
            elif df['open'][i] == df['close'][i]:
                df['intra_clf_target'][i] = 2
    
    return df_list



def create_moving_avs(df_list):
    """
    Takes in list of dfs, creates three different moving average 
    features - 10, 50, and 200 days.
    
    df_list -- list of dfs
    """
    for df in df_list:
        df['10_day_ma'] = df['close'].ewm(span = 10).mean()
        df['50_day_ma'] = df['close'].ewm(span = 50).mean()
        df['200_day_ma'] = df['close'].ewm(span = 200).mean()
     
    return df_list



def create_pct_price_changes(df_list):
    """
    Takes in a list of dfs and creates price change (%) features with various
    intervals.
    Calculates % price changes relative to the day before observed date to 
    prediction using unknown data.
    
    df_list -- list of dfs
    """
    # Creating i_day_pct_change columns in each df
    for df in df_list:
        df['1_day_pct_change'] = 0
        df['2_day_pct_change'] = 0
        df['3_day_pct_change'] = 0
        df['4_day_pct_change'] = 0
   
        # Calculating ith day price change for each col
        for i in range(5, df.shape[0]):
            # Difference in closing prices between the previous day and the previous day
            diff = float(df['close'][i-1]) - float(df['close'][i-2])
            pct = (float(diff)/float(df['close'][i-2])) * 100
            df['1_day_pct_change'][i] = pct
            
            # Difference in closing prices between the previous day and two days prior
            diff = float(df['close'][i-1]) - float(df['close'][i-3])
            pct = (float(diff)/float(df['close'][i-3])) * 100
            df['2_day_pct_change'][i] = pct
            
            # Difference in closing prices between the previous day and three days prior
            diff = float(df['close'][i-1]) - float(df['close'][i-4])
            pct = (float(diff)/float(df['close'][i-4])) * 100
            df['3_day_pct_change'][i] = pct
            
            # Difference in closing prices between the previous day and four days prior
            diff = float(df['close'][i-1]) - float(df['close'][i-5])
            pct = (float(diff)/float(df['close'][i-5])) * 100
            df['4_day_pct_change'][i] = pct
    
    return df_list



def create_intra_clf_target(df_list):
    """
    Takes in list of dfs and creates target variable for each depending
    on if the closing price for each day exceeds the opening price. If the
    closing price exceeds the opening price, target = 1.
    
    df_list -- list of dfs
    """
    for df in df_list:
        df['intra_clf_target'] = 0
        for i in range(len(df)):
            
            if df['open'][i] < df['close'][i]:
                df['intra_clf_target'][i] = 1
            
            elif df['open'][i] > df['close'][i]:
                df['intra_clf_target'][i] = 0
    
            elif df['open'][i] == df['close'][i]:
                df['intra_clf_target'][i] = 2
    
    return df_list



def time_cross_validate(df, model, param_grid):
    """
    Takes in df, model, and param_grid, and returns cross validated scores and params.
    
    df -- dataframe
    model -- model to be used for cross validation
    param_grid -- dictionary to be used for GridSearch param_grid
    """
    # Designating predictor and target columns
    predictor_cols = ['10_day_ma', 
                      '50_day_ma', 
                      '200_day_ma', 
                      '1_day_pct_change',
                      '2_day_pct_change', 
                      '3_day_pct_change', 
                      '4_day_pct_change']
    target = 'intra_clf_target'
    
    # Creating dataframe with numerical index that can be split
    dataframe = df.reset_index()

    # Creating tuples of train, test indices to be passed into the cv parameter of GridSearchCV
    train_test_indices_1 = (list(range(455)), list(range(455, 530)))
    train_test_indices_2 = (list(range(530)), list(range(530, 605)))
    train_test_indices_3 = (list(range(605)), list(range(605, 680)))
    train_test_indices_4 = (list(range(680)), list(range(680, 755)))
    
    # Splitting the df into data and target values
    data, target = dataframe[predictor_cols], df[target]
    
    # Scaling the data
    scaler = StandardScaler()
    scaler.fit_transform(data)
    
    # Instantiating Gridsearch
    clf = GridSearchCV(estimator = model,
                       param_grid = param_grid,
                       cv = [train_test_indices_1,
                             train_test_indices_2,
                             train_test_indices_3,
                             train_test_indices_4
                       ],
                       n_jobs = -1,
                       scoring = 'f1_weighted',
                       verbose = False
    )
    
    # Fitting Gridsearch
    clf.fit(data, target)
    
    # Get results and store them in a dictionary
    best_score = clf.best_score_
    best_params = clf.best_params_
    best_params['best_score'] = best_score
    
    return best_params



def time_train_test_split(df, split_date):
    """
    Takes in a df and a date to split train and test sets
    on the split date and returns X_train, X_test, y_train, y_test.
    Built to be incorporated with time_cross_validate().
    
    df -- dataframe of prices
    split_date -- str formatted as 'YYYY-MM-DD'
    """
    # Create list of split components
    train_test_sets = [] # not being used
    
    # Create train and test sets
    train = df[:split_date]
    test = df[split_date:]
    
    # Create list of columns to use in prediction, as well as target variable
    predictor_cols = ['10_day_ma', 
                      '50_day_ma', 
                      '200_day_ma', 
                      '1_day_pct_change',
                      '2_day_pct_change', 
                      '3_day_pct_change', 
                      '4_day_pct_change']
    target = 'intra_clf_target'
    
    # Create train and test sets
    X_train, X_test = train[predictor_cols], test[predictor_cols]
    y_train, y_test = train[target], test[target]
    
    return X_train, X_test, y_train, y_test



def scale_and_plant_random_forests(X_train, X_test, y_train, y_test):
    """
    Takes in train and test sets, runs a Random Forest Classifier on each,
    then returns a tuple of y test and y predicted values to be
    passed through an f1_score
    
    X_train -- training prediction data
    X_test -- testing prediction data
    y_train -- training target data
    y_test -- testing target data
    """
    # Instantiate scaler
    scaler = StandardScaler()
    
    # Fit, transform on train set, only transform test set
    scaler.fit_transform(X_train)
    scaler.transform(X_test)
    
    # Instantiate Random Forest Classifier with gridsearched parameters
    rfc = RandomForestClassifier(random_state = 1,
                                 max_depth = 7,
                                 n_estimators = 1000,
                                 criterion = 'gini',
                                 class_weight = 'balanced'
                                )
    
    # Fit to the training data
    rfc.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = rfc.predict(X_test)
    
    return (y_test, y_pred)



def scale_and_knn(X_train, X_test, y_train, y_test):
    """
    Takes in train and test sets, runs a K Nearest Neighbors Classifier on each,
    then returns a tuple of y test and y predicted values to be
    passed through an f1_score.
    
    X_train -- training prediction data
    X_test -- testing prediction data
    y_train -- training target data
    y_test -- testing target data
    """
    # Instantiate scaler
    scaler = StandardScaler()
    
    # Fit, transform on train set, only transform test set
    scaler.fit_transform(X_train)
    scaler.transform(X_test)
    
    # Instantiate K Neighbors Classifier with gridsearched parameters
    knn = KNeighborsClassifier(algorithm = 'auto',
                               leaf_size = 15, 
                               n_neighbors = 3
                              )
    
    # Fit to the training data
    knn.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = knn.predict(X_test)
    
    return (y_test, y_pred)



def compare_scores(df_list):
    """
    Takes in a list of dfs and returns a list of tuples to 
    compare the performance of two models based on their 
    respective F1 scores.
    
    df_list -- list of dfs
    """
    # Creating a list of tuples to compare model performance on each stock
    tuned_scores_comparison_list = []

    # Looping through the list of dfs
    for df in df_list:
        # Splitting train and test sets
        X_train, X_test, y_train, y_test = time_train_test_split(df, '2020-05-20')

        # Predicting with Random Forest
        (true, pred) = scale_and_plant_random_forests(X_train, X_test, y_train, y_test)

        # Evaluating Random Forest
        rfc_f1 = f1_score(true, pred, average = 'weighted')

        # Predicting with KNN
        (true, pred) = scale_and_knn(X_train, X_test, y_train, y_test)

        # Evaluating KNN
        knn_f1 = f1_score(true, pred, average = 'weighted')

        # Appending to list of tuples
        tuned_scores_comparison_list.append((rfc_f1, knn_f1))

    return tuned_scores_comparison_list



def gridsearch_arma(orders_list, train, test, n_days = 14):
    """
    Takes in a list of orders, a train set, a test set, and a number of days
    to forecast into the future. Returns a list of results for each models and
    a list of any orders that caused the model to fail.
    
    orders_list -- A list of tuples containing p and q parameters
    train -- A series of the endogenous variable
    test -- A series of the endogenous variable to measure model performance
    n_days -- An integer denoting the number of days to forecast
    """
    # Instantiating a list of results dictionaries
    results_list = []

    # Instantiating fail cache
    fail_cache = []

    # Setting number of days to forecast forward
    n_days = n_days

    # Looping through each set of parameters to find the best combination
    for o in orders_list:
        try:
            # Instantiating new dictionary of results and parameters for each gridsearched model
            results_dict = {}

            # Instantiating the ARMA model
            arma = ARMA(train, 
                        order = o, 
                   )

            # Fitting the model
            fitted_arma = arma.fit()

            # Forecasting 14 days after training data
            arma_preds = fitted_arma.forecast(steps = n_days)[0]

            # Pulling AIC score from the model
            aic = fitted_arma.aic

            # Calculating RMSE for forecasted values
            rmse = np.sqrt(mean_squared_error(test[:n_days].values, arma_preds))

            # Assigning dictionary values of results and parameters
            results_dict['order'] = o
            results_dict['aic'] = aic
            results_dict['rmse'] = rmse

            # Adding each dictionary to the list of results dictionaries
            results_list.append(results_dict) 

        except:
            fail_cache.append({'order': o,
                               })
            continue
        
    return results_list, fail_cache



def get_best_time_series(results_list):
    """
    Takes in a list of results dictionaries containing SARIMA params
    and model performance metrics. Returns best model params and scores
    for RMSE and AIC, in that order.
    
    results_list -- list of results dictionaries.
    """    
    # Sorting scores list by RMSE
    sorted_by_rmse = sorted(results_list, key = lambda x: x['rmse'])
    
    # Top model by RMSE
    top_rmse_model = sorted_by_rmse[0]
    
    # Sorting scores list by AIC
    sorted_by_aic = sorted(results_list, key = lambda x: x['aic'])
    
    # Top model by AIC
    top_aic_model = sorted_by_aic[0]
    
    return [top_rmse_model, top_aic_model]



def gridsearch_arma_multiple(df_list, orders_list, end_train_date, end_test_date, n_days = 14):
    """
    Takes in a list of dfs, a list of order params, and a number 
    of days to forecast into the future. Returns gridsearched ARMA 
    model results for each.
    
    df_list -- list of dataframes
    orders_list -- A list of tuples containing p and q parameters
    n_days -- An integer denoting the number of days to forecast
    """
    # Creating list to hold results for each df
    best_model_list = []
    
    for df in df_list:
        # Isolate endogenous variable
        df_endog = df['close']
        
        # Create differenced df to improve stationarity
        df_diff = df_endog.diff(periods = 1)
        df_diff.dropna(inplace = True)
        
        # Split df into train and test sets
        train = df_diff[:end_train_date]
        test = df_diff[end_train_date:end_test_date]

        # Calculate all models(gridsearch)
        arma_results, arma_fails = gridsearch_arma(orders_list, train, test, n_days = n_days)

        # Identify best models based on RMSE and AIC, respectively
        best_arma = get_best_time_series(arma_results)
        
        # Pull ticker from df
        ticker = df['ticker'][0]
        
        # Append ticker and model results to list of best models
        best_model_list.append([ticker, best_arma])
        
    return best_model_list



def gridsearch_sarima(orders_list, seasonal_orders_list, train, test, n_days = 14):
    """
    Takes in a list of orders, a train set, a test set, and a number of days
    to forecast into the future. Returns a list of results for each models and
    a list of any orders that caused the model to fail.
    
    orders_list -- A list of tuples containing p, d, and q parameters
    seasonal_orders_list -- A list of tuples containing p, d, q, and s parameters
    train -- A series of the endogenous variable
    test -- A series of the endogenous variable to measure model performance
    n_days -- An integer denoting the number of days to forecast
    """
    # Instantiating a list of results dictionaries
    results_list = []

    # Instantiating fail cache
    fail_cache = []

    # Setting number of days to forecast forward
    n_days = n_days

    # Looping through each set of parameters to find the best combination
    for o in orders_list:
        for so in seasonal_orders_list:
            try:
                # Instantiating new dictionary of results and parameters for each gridsearched model
                results_dict = {}

                # Instantiating the SARIMA model
                sarima = SARIMAX(train, 
                                 order = o, 
                                 seasonal_order = so, 
                                 enforce_stationarity = False,
                                 enforce_invertibility = False
                       )

                # Fitting the model
                fitted_sarima = sarima.fit()

                # Forecasting 14 days after training data
                sarima_preds = fitted_sarima.forecast(steps = n_days)

                # Pulling AIC score from the model
                aic = fitted_sarima.aic

                # Calculating RMSE for forecasted values
                rmse = np.sqrt(mean_squared_error(test[:n_days].values, sarima_preds))

                # Assigning dictionary values of results and parameters
                results_dict['order'] = o
                results_dict['seasonal_order'] = so
                results_dict['aic'] = aic
                results_dict['rmse'] = rmse

                # Adding each dictionary to the list of results dictionaries
                results_list.append(results_dict) 

            except:
                fail_cache.append({'order': o,
                                   'seasonal_order': so
                                   })
                continue
                
    return results_list, fail_cache



def gridsearch_sarima_multiple(df_list, orders_list, seasonal_orders_list, end_train_date, end_test_date, n_days = 14):
    """
    Takes in a list of dfs, a list of order params, and a number 
    of days to forecast into the future. Returns gridsearched ARMA 
    model results for each.
    
    df_list -- list of dataframes
    orders_list -- A list of tuples containing p and q parameters
    seasonal_orders_list -- A list of tuples containing p, d, q, and s parameters
    n_days -- An integer denoting the number of days to forecast
    """
    # Creating list to hold results for each df
    best_model_list = []
    
    for df in df_list:
        # Isolate endogenous variable
        df_endog = df['close']
        
        # Drop null values
        df_endog.dropna(inplace = True)
        
        # Split df into train and test sets
        train = df_endog[:end_train_date]
        test = df_endog[end_train_date:end_test_date]

        # Calculate all models(gridsearch)
        sarima_results, sarima_fails = gridsearch_sarima(orders_list, seasonal_orders_list, train, test, n_days = n_days)

        # Identify best models based on RMSE and AIC, respectively
        best_sarima = get_best_time_series(sarima_results)
        
        # Pull ticker from df
        ticker = df['ticker'][0]
        
        # Append ticker and model results to list of best models
        best_model_list.append([ticker, best_sarima])
        
    return best_model_list



def gridsearch_sarimax(orders_list, seasonal_orders_list, endog_train, 
                       exog_train, endog_test, exog_test, n_days = 14):
    """
    Takes in a list of orders, a train set, a test set, and a number of days
    to forecast into the future. Returns a list of results for each models and
    a list of any orders that caused the model to fail.
    
    orders_list -- A list of tuples containing p, d, and q parameters
    seasonal_orders_list -- A list of tuples containing p, d, q, and s parameters
    endog_train -- A series of the endogenous variable
    exog_train -- A dataframe with exogenous regressors 
    endog_test -- A series of the endogenous variable to measure model performance
    exog_test -- A dataframe of exogenous regressors for test set
    n_days -- An integer denoting the number of days to forecast
    """
    # Instantiating a list of results dictionaries
    results_list = []

    # Instantiating fail cache
    fail_cache = []

    # Setting number of days to forecast forward
    n_days = n_days

    # Looping through each set of parameters to find the best combination
    for o in orders_list:
        for so in seasonal_orders_list:
            try:
                # Instantiating new dictionary of results and parameters for each gridsearched model
                results_dict = {}

                # Instantiating the SARIMA model
                sarimax = SARIMAX(endog_train,
                                  exog_train,
                                  order = o, 
                                  seasonal_order = so, 
                                  enforce_stationarity = False,
                                  enforce_invertibility = False
                       )

                # Fitting the model
                fitted_sarimax = sarimax.fit()

                # Forecasting 14 days after training data
                sarimax_preds = fitted_sarimax.forecast(steps = n_days, exog = exog_test)

                # Pulling AIC score from the model
                aic = fitted_sarimax.aic

                # Calculating RMSE for forecasted values
                rmse = np.sqrt(mean_squared_error(endog_test[:14].values, sarimax_preds))

                # Assigning dictionary values of results and parameters
                results_dict['order'] = o
                results_dict['seasonal_order'] = so
                results_dict['aic'] = aic
                results_dict['rmse'] = rmse

                # Adding each dictionary to the list of results dictionaries
                results_list.append(results_dict) 

            except:
                fail_cache.append({'order': o,
                                   'seasonal_order': so
                                   })
                continue
                
    return results_list, fail_cache



def gridsearch_sarimax_multiple(df_list, orders_list, seasonal_orders_list, exog_vars, end_train_date, end_test_date, n_days = 14):
    """
    Takes in a list of dfs, a list of order params, and a number 
    of days to forecast into the future. Returns gridsearched ARMA 
    model results for each.
    
    df_list -- list of dataframes
    orders_list -- A list of tuples containing p and q parameters
    seasonal_orders_list -- A list of tuples containing p, d, q, and s parameters
    exog_vars -- A list of strings denoting which columns are to be used as exogenous regressors
    n_days -- An integer denoting the number of days to forecast
    """
    # Creating list to hold results for each df
    best_model_list = []
    
    for df in df_list:
        # Isolate endogenous variable
        df_endog = df['close']
        
        # Isolate exogenous variables
        df_exog = df[exog_vars]
        
        # Drop null values
        df_endog.dropna(inplace = True)
        df_exog.dropna(inplace = True)
        
        # Split endog df into train and test sets
        endog_train = df_endog[:end_train_date]
        endog_test = df_endog[end_train_date:end_test_date]
        
        # Split exog df into train and test sets
        exog_train = df_exog[:end_train_date]
        exog_test = df_exog[end_train_date:end_test_date]

        # Calculate all models(gridsearch)
        sarimax_results, sarimax_fails = gridsearch_sarimax(orders_list, seasonal_orders_list, endog_train, 
                                                            exog_train, endog_test, 
                                                            exog_test, n_days = n_days)

        # Identify best models based on RMSE and AIC, respectively
        best_sarimax = get_best_time_series(sarimax_results)
        
        # Pull ticker from df
        ticker = df['ticker'][0]
        
        # Append ticker and model results to list of best models
        best_model_list.append([ticker, best_sarimax])
        
    return best_model_list



def split_sequence(sequence, n_steps_in, n_steps_out):
    """
    Split a sequence into samples
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



def rnn_multiple(df_list, n_steps_in, n_steps_out, end_train_date, end_test_date):
    """
    Takes in a list of dataframes, number of input and output steps, 
    and dates to end train and test sets.
    
    df_list -- list of dataframes
    n_steps_in -- integer
    n_steps_out -- integer
    end_train_date -- string in 'YYYY-MM-DD' format where the date field is optional
    end_test_date -- string in 'YYYY-MM-DD' format where the date field is optional
    """
    # Create list of results to store scores and fail cache to store failed dfs
    results_list = []
    fail_cache = []
    
    for df in df_list:
        try:
            # Define a dictionary to store model performance on each stock
            result = {}

            # Get ticker and add to result dictionary
            result['ticker'] = df['ticker'][0]

            # Remove exogenous regressors
            df = df['close']

            # Endogenous regressors train/test split
            rnn_train = df[:end_train_date]
            rnn_test = df[end_train_date:end_test_date]

            # Setting raw input sequence
            raw_seq = rnn_train.values

            # Reshaping raw_seq from one dimensional to two dimensional vector for scaling
            raw_seq = raw_seq.reshape(raw_seq.shape[0], 1)

            # Setting number of input and output/forecast steps
            # 'From the past n_steps_in trading days, predict the next n_steps_out'
            n_steps_in, n_steps_out = 100, 14

            # Splitting the sequence
            X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

            # Number of features is one because time series is univariate
            n_features = 1

            # Reshape from (samples, timesteps) into (samples, timesteps, features)
            X = X.reshape(X.shape[0], X.shape[1], n_features)

            # Define model architecture
            model = Sequential()
            model.add(LSTM(75, input_shape = (n_steps_in, n_features)))
            model.add(Dropout(.1))
            model.add(Dense(25))
            model.add(Dropout(.1))
            model.add(Dense(n_steps_out))
            model.compile(optimizer = 'adam', loss = 'mse')

            # Instantiate History object to plot loss
            history = History()

            # Fit the model
            model.fit(X, y, epochs = 250, verbose = False, shuffle = False, callbacks = history)

            # Set train evaluation input and outputs
            X_train = rnn_train.values[-(n_steps_in+n_steps_out):-n_steps_out]
            y_train = rnn_train.values[-n_steps_out:]

            # Reshape input
            X_train = X_train.reshape(1, n_steps_in, n_features)

            # Set test evaluation input and outputs
            X_test = rnn_train.values[-n_steps_in:]
            y_test = rnn_test.values

            # Reshape input 
            X_test = X_test.reshape(1, n_steps_in, n_features)

            # Predicting on train and test sets
            y_pred_train = model.predict(X_train, verbose = False)[0]
            y_pred_test = model.predict(X_test, verbose = False)[0]
            # [0] is to access nested array
            
            # Store test predictions in result dictionary
            result['y_pred_test'] = y_pred_test
            
            # Calculate RMSE for train and test predictions
            result['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
            result['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred_test))

            # Define loss variable for plotting
            result['loss'] = history.history['loss']

            # Append result dictionary to results_list
            results_list.append(result)
            
        except:
            fail_cache.append(df['ticker'][0])
            continue
            
    return results_list, fail_cache 