"""This python file is a collection for data cleaning"""

import pandas as pd
import linear_regression as r

def drop_feature(data, features):
    return data.drop(columns= features)

def column_log(data, features):
    data_log = data.loc[features].applymap(np.log).add_suffix('_log') 
    return data_log



def import_admissions_dataset():
    admissions_dataset = pd.read_csv('data/Admission_Predict_Ver1.1.csv', header=0)

    #rename columns with spaces at the end
    admissions_dataset.rename(columns = {'LOR ':'LOR','Chance of Admit ':'Chance of Admit'},inplace=True)

    #remove spaces from columns
    r.edit_column_names(admissions_dataset, inplace=True, lower=False).head()
    
    return admissions_dataset