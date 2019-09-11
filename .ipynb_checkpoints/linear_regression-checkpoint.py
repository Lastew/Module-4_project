
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