
# import python packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

# import sklearn linear models

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, r2_score

# %matplotlib inline



# import relevant libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import datetime
import matplotlib.pyplot as plt #better done in notebook
# %matplotlib inline #must be done in notebook

def help_tools(library_name = "rs."):
    """print names of relevant tools"""

    tools_not_included = [
    "sns.pairplot(df)"
    ]

# def drop_feature(data, features):
#     return data.drop(columns= features)

# change column names
# admissions_dataset.rename(columns = {'LOR ':'LOR','Chance of Admit ':'Chance of Admit'},inplace=True)

#function to create logs
#use X_train_log = X_train.loc[:].applymap(np.log).add_suffix('_log')

# add to scatter_matrix
# cov() correlation coefficient
# corr() correlation coefficient

# add df.drop_duplicates(subset=['ID'], inplace=True)
# add df.dropna(subset=['Release Clause'],inplace=True)

#resample whole df
#stocks_month_df = stocks_df.resample('m').mean()

#resample series
# bus_ridership_annual = df.bus_ridership.resample('y').sum()
# bus_ridership_annual = pd.DataFrame(bus_ridership_annual,columns=['bus_ridership'])
# bus_ridership_annual[:-1].plot()
# plt.show()

#line plot
# stocks_month_df.plot()
# plt.show()

    cleaning_tools = [
    "edit_column_names(df_original, inplace=True, lower=True):"
    "find_zeroes(df):",
    "find_null(df):",
    "check_uniques(df):",
    "drop_feature(df, features):"
    ]

    regression_tools = [
    "train_test_split_fxn(X, y, random_state = 12345, test_size = .25):",  #update with parameters
    "pair_plot(df):",
    "create_log(df,features):",
    "statsmodel_ols_formula_regression(X,y,target_name=False,features_list=False,suppress=False,show_resid=False,qqplot_line='s'): #preferred over sm"
    ]

    timeseries_tools = [
    "seasonal_decomposition(timeseries,freq=12):",
    "test_stationarity(timeseries, window=12):",
    "acf_pacf(timeseries, start=0, lags=18):"
    ]

    #add the library name prior to printing
    print("\n")
    print_tools("useful tools not included:",tools_not_included,library_name)
    print_tools("cleaning tools:",cleaning_tools,library_name)
    print_tools("regression tools:",regression_tools,library_name)
    print_tools("time series tools:",timeseries_tools,library_name)



def print_tools(title,tool_list,library_name):
    """
    print tools with the requisite formatting
    [title]: [library_name].tool_name for each tool in tool in list
    line break
    """
    print(title,*prepend(tool_list,library_name),sep="\n")
    print("\n")

def prepend(list, str):
    """
    Python3 program to insert the string
    at the beginning of all items in a list
    """
    #add a {} at the end of the str
    str += '{0}'
    #recreate the list starting with the str
    list = [str.format(i) for i in list]
    return(list)

def edit_column_names(df_original, inplace=True, lower=True):
    """cleans the column names to only be lowercase text and underscores"""
    if inplace:
        df = df_original
    else:
        df = df_original.copy()

    # check what this does. We may want later
    df.columns = df.columns.str.replace('[^a-zA-Z0-9\s+\_]+','', regex=True)

    # replace " " with underscores
    df.columns = df.columns.str.replace('\s+', '_',regex=True)

    if lower:
        df.columns = df.columns.str.lower()
    return df


def find_null(df):
    """
    output various tools to find null
    """

    print("more find_null tools needed")

# import seaborn for sns.heatmap
import seaborn as sns
def find_zeros(df):
    """
    output various tools to find zeros
    """
    sns.heatmap(df==0)
    plt.show()

    print("more find_zero tools needed")

def check_uniques(df):
    """
    output various tools to check number of unique in each feature
    """
    features = list(df.columns)
    features = [
    "{}: {}".format(feature,df[feature].nunique())
    for feature
    in features]
    print(*features,sep="\n")

def drop_feature(df, features):
    return df.drop(columns= features)


# import train_test_split
from sklearn.model_selection import train_test_split
def train_test_split_fxn(X, y, random_state = 12345, test_size = .25):
    """
    parameters:
    X - df of parameters
    y - series or df of target
    randome state - optional
    test_size - proportion of df
    output: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = 0,0,0,0
    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, random_state = 42, test_size = test_size)
    return X_train, X_test, y_train, y_test

import seaborn as sns
def pair_plot(df):
    sns.pairplot(df)
    plt.show()

def create_log(df,features):
    df_new = df
    # add functions to turn the feature or feature list into log of features
    # save each new feature as feature_log
    print("pending completion")
    return df_new

def get_statsmodel_formula_string(target_name, features_list):
    formula = False
    return formula


import sklearn.metrics as metrics
def regression_results(y_true, y_pred):
    """
    prints regression results for any y_true and y_pred
    """

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    print('\n')

# import ols from statsmodels.formula.api
from statsmodels.formula.api import ols
def statsmodel_ols_formula_regression(X,y,target_name=False,features_list=False,suppress=False,show_resid=False,qqplot_line='s'): #preferred over sm
    """
    This model includes an intercept and seems to be better suited to
    a running an Ordinary LS than the 'statsmodels.api sm regression'
    print: model summary2s
    output: model
    """
    # assert target_name,"missing target_name"

    # set target_name if not already set
    if not target_name:
        target_name = 'target'
    # create features_list if not already given
    if not features_list:
        features_list = list(X.columns)

    #create dataframe for regression
    df = X.copy()
    df[target_name]=y

    formula = '{target_name}~{features_list}'.format(
        target_name=target_name,
        features_list= "+".join(features_list)
        )

    print("\nformula: {}".format(formula),"\n")

    model = ols(formula=formula, data = df).fit()
    if not suppress:
        print(model.summary())
    if show_resid:
        res = model.resid
        fig = sm.qqplot(res, line=qqplot_line)
        plt.show()
    return model

# def show_resid()


from sklearn.linear_model import LinearRegression
def sklearn_ols_regression(X,y,print_coefficients=True,print_resid=False,show_resid=False,qqplot_line='s'):
    """
    ols regression in sklearn
    print: coefficients (optional), regression metrics (optional), qqplot (optional)
    output: SKlearn LinearRegression object
    """
    # initialize a linear regression model in sklearn
    linrig = LinearRegression()
    # fit linear model to training data
    linrig.fit(X, y)
    y_pred = linrig.predict(X)

    if print_coefficients:
        print('Features: ', list(X.columns))
        print('Coefficients: ', linrig.coef_)
        print('y-intercept: ', np.round(linrig.intercept_,3))
        print('\n')
#         print('R^2: ', np.round(linrig.score(X,y), 3)) # this is the r squared value from sklearn

    if print_resid:
        regression_results(y, y_pred)
#         print('MSE: ', mean_squared_error(y, y_pred, multioutput='raw_values'))
    if show_resid:
        sk_res = pd.Series(data=[np.abs(y - y_pred)])
        #correct this later
        print('QQPLOT OF RESID NOT WORKING. IS RESID INCORRECT OR WRONG ORDER?\n')
        fig = sm.qqplot(sk_res,line=qqplot_line)
        plt.show()
    return linrig



# import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
#function for seasonal decomposition
def seasonal_decomposition(timeseries,freq=12):
    decomposition = seasonal_decompose(timeseries,freq=12)
    fig = plt.figure()
    fig = decomposition.plot()
    fig.set_size_inches(15, 8)
    plt.show()
    return decomposition

#plot acg and pacf
def acf_pacf(timeseries, start=0, lags=18):
    """
    output: visualization of correlograms
    Autocorrelogram & Partial Autocorrelogram
    are useful to estimate each models autocorrelation.
    """

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    #plot the ACF
    fig = sm.graphics.tsa.plot_acf(timeseries.iloc[start:], lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    #plot the PACF
    fig = sm.graphics.tsa.plot_pacf(timeseries.iloc[start:], lags=lags, ax=ax2)

#import adfuller for Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller
#function that will help us to quickly run Dickey-Fuller Test
def test_stationarity(timeseries, window=12,dropna=True):
    """
    plot rolling mean and std, Run Dickey-Fuller Test
    output: NO VALUE RETURNED, plot rolling mean and std; print Dickey-Fuller
    """
    timeseries.dropna(inplace=dropna)

    #Determing rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries.iloc[window:], color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


    
    

def inverse_transform(y,target):
    y = y.loc[:,[target]].apply(lambda x : 1/x).add_suffix('_inv_transform')
    return y

def get_residual_plot(X_col,y,rows=1,cols=1,position=1,ax=False,title=False):
    plt.subplot(rows,cols,position)
    sns.residplot(X_col,y)
    if title:
        ax = plt.xlabel(title)

def residual_plot(X,y,feature_list=False,cols=1,savefig=False,save_path=False,title="Distribution Plot",**kwargs):
    """
    residual plot for multiple features
    parameters: **kwargs passed to sns.plt.subplots()
    savefig allows to save residual plot
    save_path must be identified
    title: set to False to remove
    """
    
    if not feature_list:
        feature_list = list(X.columns)
    
    # residual plot for all three features

    num_plots = len(feature_list)
    rows = num_plots // cols
    f, axs = plt.subplots(rows,cols,figsize=(10,10),**kwargs)
    
    y1 = y.copy()
    
    for i,feature in enumerate(feature_list):
        get_residual_plot(X[feature],y1,rows=rows,cols=cols,position=i+1,ax=axs[i],title=feature)
        
        # need to include something like the below to allow for multiple cols
#         row_position = (i+1) % cols
#         col_position = ((i) // cols)+1
#         get_residual_plot(X[feature],y1,rows=rows,cols=cols,position=i+1,ax=axs[row_position,col_position],title=feature)

    if title:
        f.suptitle(title, fontsize=16)
    if savefig:
        if save_path:
            plt.savefig(save_path)
        else:
            print("please specify path to save image")
    plt.show()
        