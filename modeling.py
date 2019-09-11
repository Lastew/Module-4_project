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
import itertools

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

# %matplotlib inline


# a formula to calculate adjusted r squared

def adjusted_r_suared(r_squared, num_samples, num_regressors):
    return 1 - ((1-r_squared)*(num_samples - 1) / (num_samples - num_regressors - 1))


def lasso_test_scores(X_list,y):
    degrees = [1,2,3]
    alphas = [0.0001,0.01,0.1,0.5,1,2,5,10,30,100] 
    results = []

    #create 
    for degree, alpha in itertools.product(degrees, alphas):
        y_lasso_test_mean = model_expreriment(
            X_list[degree-1],
            y,
            num_iter = 5,
            alpha = alpha,
            max_iter= 1000,
            show_plot = False
        )
        results.append([degree, alpha, y_lasso_test_mean])

    r2_test = pd.DataFrame(results, columns=['degrees', 'alpha', 'r2'])
    return r2_test

def get_lasso_model_list(X_list,y,alpha=1,max_iter=10000):
    """
    Generate a list of lasso models corresponding to a list of X dataframes
    """
    lasso_model_list = [
        r.get_lasso(X[i],y,alpha=0.00001,max_iter=10000).score(X[i],y)
        for i
        in range(0,len(X))
    ]
    return lasso_model_list


def get_lasso_model_score(X_list,y,alpha=1,max_iter=10000):
    """
    Generate a list of lasso models corresponding to a list of X dataframes
    """
    lasso_model_list = [
        r.get_lasso(X[i],y,alpha=0.00001,max_iter=10000).score(X[i],y)
        for i
        in range(0,len(X))
    ]
    return lasso_model_list

def model_expreriment(X,y,num_iter = 5,alpha = 1, max_iter= 1000,show_plot = False):
    """
    get the mean lasso.score(X_test,y_test) for any 
    """
    y_lasso_test = []
    for i in range(num_iter):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

        ## Standard scale mean = 0, variance = 1
        sd = StandardScaler()

        sd.fit(X_train)

        X_train = sd.transform(X_train)

        X_test = sd.transform(X_test)

        lasso = Lasso(alpha = alpha, max_iter= max_iter)
        lasso.fit(X_train, y_train)
                        
        y_lasso_test.append(lasso.score(X_test, y_test))

        i+=1

    if show_plot:
        plt.plot(y_lasso_test, label = 'lasso')
        plt.ylim([0,1])
        plt.ylabel('R2 test score')
        plt.xlabel('number of iterations')
        plt.legend()
    return (sum(y_lasso_test)/len(y_lasso_test))