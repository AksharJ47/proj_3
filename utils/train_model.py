"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Team_2_DBN

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
from sklearn.model_selection import train_test_split 
import lightgbm as lgb

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/Train_Zindi.csv')
riders = pd.read_csv('data/Riders_Zindi.csv')
test_df = pd.read_csv('data/Test_Zindi.csv')
test = test_df.copy()
train_df = train.copy()
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

# Assume 0mm of rain where precipitation is missing
# Impute missing temperature with average
def Impute(input_df):
    '''Function fills missing values on the Temperature and
       Precipitation columns, all missing temperatures are
       imputed with the average temperature while all
       Precipitation columns are filled with 0mm of rain
    '''
    df = input_df.copy()
    cols_to_impute = ['Temperature',
                      'Precipitation(mm)']
    for col in cols_to_impute:
        if col == 'Temperature':
            a = round(df[col].mean(),1)
        if col == 'Precipitation(mm)':
            a = round(df[col].mean(),1)
        df[col] = df[col].fillna(a) 
    return (df)

train = Impute(train)

# Time change function

def time_change(input_df):
    '''Converts time format %H:%M:%S to seconds past midnight(00:00) of
       the same day rounded to the nearest second.
       ------------------------------
       12:00:00 PM --> 43200
       01:30:00 AM --> 5400
       02:35:30 PM --> 9330
     '''
    df = input_df.copy()
    from pandas.api.types import is_numeric_dtype
    def time_fn(row):
        b = row.split(' ')
        if b[1] == 'AM':
            c = 0
        else:
            c = 12
        b = b[0].split(':')
        b = [int(i) for i in b]
        if b[0] == 12:
            c -= 12
        # convertion to hours
        b[0] = (b[0] + c)*3600
        b[1] = (b[1])*60.0
        b[2] = (b[2])
        row = int(sum(b))
        return(row)
    time_columns = [
                'Pla_Time',\
                'Con_Time',\
                'Arr_Pic_Time',\
                'Pickup_Time',\
               ]
    for col in df.columns:
        if col in time_columns:
            if is_numeric_dtype(df[col]) is False:
                df[col] = df[col].apply(lambda x: time_fn(x))
            else:
                pass
    return(df)

train = time_change(train)


# Add ride experience column
train['Rider_Exp'] = pd.qcut(train['Age'],
                                       q=[0, .25, .75, 1],
                                       labels=['low', 'medium', 'high'])

# Create Temperature band Column - 3 categories - low, mid, high
train['Temp_Band'] = pd.qcut(train['Temperature'],
                                       q=[0, .25, .75, 1],
                                       labels=['low', 'medium', 'high'])

# Time difference

def time_diffs(input_df):
    df = input_df.copy()
    df['Conf_Pla_dif'] = df['Con_Time'] - df['Pla_Time']
    df['Arr_Con_dif'] = df['Arr_Pic_Time'] - df['Con_Time']
    df['Pic_Arr_dif'] = df['Pickup_Time'] - df['Arr_Pic_Time']

    return df

train = time_diffs(train)


# Create manhattan dist
def manhattan(input_df):
    '''Calculates the manhattan distance between two location given
       the longitude and latitude of the locations
    '''
    df = input_df.copy()
    a = np.abs(df['Pickup_Lat'] - df['Destination_Lat'])
    b = np.abs(df['Pickup_Lon'] - df['Destination_Lon'])
    df['manhattan_dist'] = a + b
    return (df)

train = manhattan(train)

# Haversine distance

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
                                                           lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h
def add_haversine(input_df):
    input_df_1 = input_df.copy()
    input_df_1['distance_haversine'] = haversine_array(
                                    input_df_1['Pickup_Lat'].values,
                                    input_df_1['Pickup_Lon'].values,
                                    input_df_1['Destination_Lat'].values,
                                    input_df_1['Destination_Lon'].values)
    return input_df_1


train = add_haversine(train)

# Encode Rider Exp,Temp_Band and Personal/Business and Normalise all features
def encode_normalize(input_df):
    from pandas.api.types import is_numeric_dtype
    df = input_df.copy()
    to_encode = ['Rider_Exp',
                 'Personal_Business',
                 'Temp_Band']
    for col in (df.drop(to_encode, axis=1).columns):
        if is_numeric_dtype(df[col]) and col not in to_encode and col != "Time_Pic_Arr":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df[[col]] = scaler.fit_transform(df[[col]])             
    df = pd.get_dummies(df, columns=to_encode, drop_first=True)
    return(df)

train = encode_normalize(train)

# Extract feature columns
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


# data_df = data_encoded_df[numeric_cols]
y = train[:len(train_df)]['Time_Pic_Arr']
X = train[numeric_cols][:len(train_df)]
test = train[numeric_cols][len(train_df):]

# Setting a Random State

rs = 42

# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state = rs)


lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Training model on optimal parameters

lparams = {
           'learning_rate': 0.1, 'min_data_in_leaf': 300, 
           'n_estimators': 75, 'num_leaves': 20, 'random_state':rs,
           'objective': 'regression', 'reg_alpha': 0.02,
          'feature_fraction': 0.9, 'bagging_fraction':0.9}


lgbm = lgb.train(lparams, lgb_train, valid_sets=lgb_eval, num_boost_round=20,
                 early_stopping_rounds=20)



# Pickle model for use within our API

save_path = 'assets/trained-models/pickle_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lgbm, open(save_path,'wb'))



