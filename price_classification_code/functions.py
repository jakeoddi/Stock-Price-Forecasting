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