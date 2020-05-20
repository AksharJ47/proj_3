"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# All Dependencies we might need

import pickle
import pandas as pd
import math
import numpy as np
from dateutil.parser import parse
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm.sklearn import LGBMRegressor



# Fetch training data and preprocess for modeling
train = pd.read_csv('data/Train_Zindi.csv')
riders = pd.read_csv('data/Riders_Zindi.csv')

# Filter out some riders values 

#riders_new = riders[riders['No_Of_Orders'] > 350]
riders = riders[riders['Average_Rating'] > 12.2]

# Merge the two dataframes

train = train.merge(riders, how='left', on='Rider Id')# Equivalent to query

# Drop out some columns ( Logic supplied on the notebook )

train = train.drop(['Vehicle Type','Precipitation in millimeters' ,
                    'User Id','Order No','Rider Id' ,'Platform Type',
                    'Personal or Business',
                    'Arrival at Destination - Day of Month',
                    'Arrival at Destination - Weekday (Mo = 1)',
                    'Arrival at Destination - Time'],axis = 1)

train = train.fillna(train.mean()) # Populate all nulls

# Time manipulation functions

def time_parse_hr(time) :
    dt = parse(time)
    hour = dt.hour    
    return hour

def time_parse_min(time) :
    dt = parse(time)
    minute = dt.minute   
    return minute

def time_parse_sec(time) :
    dt = parse(time)
    second = dt.second    
    return second

# # Time manipulation application

train['Placement - Hour'] = train['Placement - Time'].apply(time_parse_hr)
train['Placement - Minute'] = train['Placement - Time'].apply(time_parse_min)
train['Placement - Second'] = train['Placement - Time'].apply(time_parse_sec)
                                                            
train['Confirmation - Hour'] = train['Confirmation - Time'
                                     ].apply(time_parse_hr)
train['Confirmation - Minute'] = train['Confirmation - Time'
                                     ].apply(time_parse_min)
train['Confirmation - Second'] = train['Confirmation - Time'
                                     ].apply(time_parse_sec)

train['Arrival at Pickup - Hour'] = train['Arrival at Pickup - Time'
                                        ].apply(time_parse_hr)
train['Arrival at Pickup - Minute'] = train['Arrival at Pickup - Time'
                                          ].apply(time_parse_min)
train['Arrival at Pickup - Second'] = train['Arrival at Pickup - Time'
                                          ].apply(time_parse_sec)

train['Pickup - Hour'] = train['Pickup - Time'].apply(time_parse_hr)
train['Pickup - Minute'] = train['Pickup - Time'].apply(time_parse_min)
train['Pickup - Second'] = train['Pickup - Time'].apply(time_parse_sec)


# Drop original times

train = train.drop(['Placement - Time' , 'Confirmation - Time',
                 'Arrival at Pickup - Time' ,'Pickup - Time'],
                 axis = 1)

# Filter outliars ( See notebook for logic )

train['Time from Pickup to Arrival'] = np.log(train[
                     'Time from Pickup to Arrival'])

train = train[train['Time from Pickup to Arrival'] > 5.1]

train['Time from Pickup to Arrival'] = np.exp(train[
                             'Time from Pickup to Arrival'])

train = train[train['Time from Pickup to Arrival'] < 5300 ]

# Re-Order Columns for uniformity ( Hard coded in the order of the notebook)

columns_to_keep = ['Distance (KM)','Destination Lat','Confirmation - Second',
 'Pickup - Minute','No_Of_Orders','Pickup Long','Placement - Second',
'Arrival at Pickup - Minute','Pickup Lat','Arrival at Pickup - Second',
 'No_of_Ratings','Placement - Minute','Pickup - Second','Average_Rating',
 'Confirmation - Minute','Destination Long','Age','Confirmation - Day of Month',
 'Pickup - Hour','Confirmation - Hour','Placement - Day of Month',
 'Arrival at Pickup - Day of Month','Temperature','Pickup - Day of Month',
 'Placement - Hour','Confirmation - Weekday (Mo = 1)','Placement - Weekday (Mo = 1)',
 'Arrival at Pickup - Weekday (Mo = 1)','Pickup - Weekday (Mo = 1)',
 'Arrival at Pickup - Hour']


# Regression

X = train[columns_to_keep].values
y = train['Time from Pickup to Arrival'].astype(int)



# Create a train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.80, test_size = 0.2, random_state=15)


# Fit model

lm_regression = LGBMRegressor(num_leaves = 23 ,random_state = 15 ,
                              learning_rate = 0.1 )
print ("Training Model...")
lm_regression.fit(X_train, y_train)

# Pickle model for use within our API

save_path = 'assets/trained-models/sendy_simple_lm_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))




