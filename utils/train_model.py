"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Install dependecies

# pip install lightgbm
# pip install xgboost

# Import Libraries

import pickle
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV 
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import lightgbm as lgb



''' Data cleaning and formating '''

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/Train_Zindi.csv')
riders = pd.read_csv('data/Riders_Zindi.csv')
test_df = pd.read_csv('data/Test_Zindi.csv')
test = test_df.copy()
# Drop data not available in test, Pickup Time + label = Arrival times

train = train.drop(['Arrival at Destination - Day of Month',
                          'Arrival at Destination - Weekday (Mo = 1)',
                          'Arrival at Destination - Time'], axis=1)

# Combine train & test & riders to create a complete df

test['Time from Pickup to Arrival'] = [np.nan]*test.shape[0]
full_df = pd.concat([train, test], axis=0, ignore_index=True)
train = pd.merge(full_df, riders, how='left', left_on='Rider Id',
                          right_on='Rider Id', left_index=True)


# Variable name cleaning and re-formating

feature_names = {"Order No": "Order_No", "User Id": "User_Id",
                 "Vehicle Type": "Vehicle_Type",
                  "Personal or Business": "Personal_Business",
                   "Placement - Day of Month": "Pla_Mon",
                    "Placement - Weekday (Mo = 1)": "Pla_Weekday",
                      "Placement - Time": "Pla_Time", 
                  "Confirmation - Day of Month":"Con_Day_Mon",
                  "Confirmation - Weekday (Mo = 1)": "Con_Weekday",
                  "Confirmation - Time": "Con_Time", 
                  "Arrival at Pickup - Day of Month": "Arr_Pic_Mon",
                 "Arrival at Pickup - Weekday (Mo = 1)": "Arr_Pic_Weekday", 
                "Arrival at Pickup - Time": "Arr_Pic_Time",
                "Platform Type": "Platform_Type",
                 "Pickup - Day of Month": "Pickup_Mon",
                "Pickup - Weekday (Mo = 1)": "Pickup_Weekday",           
                "Pickup - Time": "Pickup_Time",
                 "Distance (KM)": "Distance(km)",
                 "Precipitation in millimeters": "Precipitation(mm)",
               "Pickup Lat": "Pickup_Lat", "Pickup Long": "Pickup_Lon", 
               "Destination Lat": "Destination_Lat",
               "Destination Long":"Destination_Lon",
               "Rider Id": "Rider_Id",
               "Time from Pickup to Arrival": "Time_Pic_Arr"}

train = train.rename(columns=feature_names)

# Function to convert time to seconds after midnight 

def time_conv(input_df):
    input_df_1 = input_df.copy()
    def timetosecs(x):
        if len(x) == 10:
            if x[-2:] == 'AM':
                x = (float(x[0])*3600) + (float(x[2:4])*60) + float(x[5:7])
            else:
                x = (float(x[0])*43200) + (float(x[2:4])*60) + float(x[5:7])
        else:
            if x[-2:] == 'AM':
                x = (float(x[0:2])*3600) + (float(x[3:5])*60) + float(x[6:8])
            else:
                x = (float(x[0:2])*43200) + (float(x[3:5])*60) + float(x[6:8])
        return x
    
    input_df_1['Pla_Time'] = input_df_1['Pla_Time'].apply(timetosecs)
    input_df_1['Con_Time'] = input_df_1['Con_Time'].apply(timetosecs)
    input_df_1['Arr_Pic_Time'] = input_df_1['Arr_Pic_Time'].apply(timetosecs)
    input_df_1['Pickup_Time'] = input_df_1['Pickup_Time'].apply(timetosecs)
    
    return input_df_1

# Implement time conversion function

train = time_conv(train)

# Function to add Columns for time differences

def time_diffs(input_df):
    time_diffs_df = input_df.copy()
    time_diffs_df['Conf_Pla_dif'] = time_diffs_df['Con_Time'
                                                ] - time_diffs_df['Pla_Time']
    time_diffs_df['Arr_Con_dif'] = time_diffs_df['Arr_Pic_Time'
                                                 ] - time_diffs_df['Con_Time']
    time_diffs_df['Pic_Arr_dif'] = time_diffs_df['Pickup_Time'
                                            ] - time_diffs_df['Arr_Pic_Time']
    
    return time_diffs_df

train = time_diffs(train)

''' Feature Engineering and Selection '''


# Add Rider Experience based on Age Column - Low - Medium - High



train['Rider_Exp'] = pd.qcut(train['Age'], q=[0, .25, .75, 1],
                                    labels=['low', 'medium', 'high'])

# Filling Missing Values for Temperature and Precipitation - used the Mean


train['Temperature'] = train['Temperature'].fillna(
                                train['Temperature'].mean())


train['Precipitation(mm)'].fillna(train[
                         'Precipitation(mm)'].mean(), inplace=True)

## Create Temperature band Column - 3 categories - low, mid, high

train['Temp_Band'] = pd.qcut(train['Temperature'],
                        q=[0, .25, .75, 1], labels=['low', 'medium', 'high'])


# Manhattan distance function

def manhattan_distance(lat1, lng1, lat2, lng2):
    a = np.abs(lat2 -lat1)
    b = np.abs(lng1 - lng2)
    return a + b

# Function to add Manhattan to DF

def added_manhattan(input_df):
    input_df_1 = input_df.copy()
    input_df_1['distance_manhattan'
               ] = manhattan_distance(input_df_1['Pickup_Lat'].values,
                                      input_df_1['Pickup_Lon'].values,
                                      input_df_1['Destination_Lat'].values,
                                      input_df_1['Destination_Lon'].values)
    return input_df_1


# Implement manhatten distance function

train = added_manhattan(train)

# Haversine distance function

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
                                                       lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    
    return h

# Function to add haversine distance

def add_haversine(input_df):
    input_df_1 = input_df.copy()
    input_df_1['distance_haversine'
               ] = haversine_array(input_df_1['Pickup_Lat'].values,
                                  input_df_1['Pickup_Lon'].values,
                                  input_df_1['Destination_Lat'].values,
                                  input_df_1['Destination_Lon'].values)
                                   
    return input_df_1

# Implementing harvisine distance function

train = add_haversine(train)

# This is to check if there is any difference between
# the columns with Days of Month or Weekday of Month

month_cols = [col for col in train.columns if col.endswith('Mon')]
weekday_cols = [col for col in train.columns if col.endswith('Weekday')]

count = 0
instances_of_different_days = [];
for i, row in train.iterrows():
    if len(set(row[month_cols].values)) > 1:
        # print(count+1, end='\r')
        count = count + 1
        instances_of_different_days.append(list(row[month_cols].values))
        
               
# Drop columns based on:
# Days of Month or Weekday of Month are the same except for 2 rows.
# The delivery service is same day
# All Vehicle types are Bikes, Vehicle Type is not necessary.

train['Day_of_Month'] = train[month_cols[0]]
train['Day_of_Week'] = train[weekday_cols[0]]

train.drop(month_cols+weekday_cols, axis=1, inplace=True)
train.drop('Vehicle_Type', axis=1, inplace=True)

#Convert Personal_Business Temp_Band using LabelEncoding

le = LabelEncoder()
le.fit(train['Personal_Business'])
train['Personal_Business'] = le.transform(
                                train['Personal_Business'])

# train['Personal_Business'][:2]

# Rider_Exp convert Label Encoding

le.fit(train['Rider_Exp'])
train['Rider_Exp'] = le.transform(train['Rider_Exp'])
# time_conv_df['Rider_Exp'][:2]

# Convert Temp_Band using LabelEncoding

le.fit(train['Temp_Band'])
train['Temp_Band'] = le.transform(train['Temp_Band'])
# time_conv_df['Temp_Band'][:2]

# This function splits Columns into Data types
# this makes it easier to select & plot numeric features
# against the Target Variable

numeric_cols = []
object_cols = []
time_cols = []
for k, v in train.dtypes.items():
    if (v != object):
        if (k != "Time_Pic_Arr"):
            numeric_cols.append(k)
    elif k.endswith("Time"):
        time_cols.append(k)
    else:
        object_cols.append(k)
        

## Feature Selection & Dropping of the Target Variable

features = numeric_cols 

data_df = train[features]
train_end = 21201
y = train[:train_end]['Time_Pic_Arr']
train = data_df[:train_end]             # Train and Test are redifined
test = data_df[train_end:]

''' 
    Note that here we will skip Cross-Validation and Grid search steps
    to increase the time response of the training
    
    These steps can be found in the notebook
    
'''

'''
    Regression Implementation 
'''

## Splitting the Data into Train & Test sets

rs = 42

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2,
                                            shuffle=True,random_state = rs)              # No random state

# Training the model on best paramters

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

 
lparams = {'learning_rate': 0.1, 'min_data_in_leaf': 300, 
           'n_estimators': 75, 'num_leaves': 20, 'random_state':rs,
           'objective': 'regression', 'reg_alpha': 0.02,
          'feature_fraction': 0.9, 'bagging_fraction':0.9}


print ("Training Model...")
# lm_regression.fit(X_train, y_train)

lgbm = lgb.train(lparams, lgb_train, valid_sets=lgb_eval, num_boost_round=20,
                 early_stopping_rounds=20 )

# Pickle model for use within our API

save_path = 'assets/trained-models/sendy_simple_lm_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lgbm, open(save_path,'wb'))


