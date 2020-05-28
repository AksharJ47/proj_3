"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder


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


# Import train for easy pre-processing

train = pd.read_csv('utils/data/Train_Zindi.csv')

# Define order to facilitate merge

order = ['Order No', 'User Id', 'Vehicle Type', 'Platform Type',
       'Personal or Business', 'Placement - Day of Month',
       'Placement - Weekday (Mo = 1)', 'Placement - Time',
       'Confirmation - Day of Month', 'Confirmation - Weekday (Mo = 1)',
       'Confirmation - Time', 'Arrival at Pickup - Day of Month',
       'Arrival at Pickup - Weekday (Mo = 1)', 'Arrival at Pickup - Time',
       'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)', 'Pickup - Time',
       'Distance (KM)', 'Temperature', 'Precipitation in millimeters',
       'Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long',
       'Rider Id', 'Time from Pickup to Arrival', 'No_Of_Orders', 'Age',
       'Average_Rating', 'No_of_Ratings']


# ---------------------------------------------------------------------------#

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    
    # Add Target in the test dataset ( already merged with riders )
    
    feature_vector_df['Time from Pickup to Arrival'] = [np.nan
                                            ]*feature_vector_df.shape[0]

    feature_vector_df = feature_vector_df[order]
    
    predict_vector = feature_vector_df.copy()
    
    # Concatinate train and test(which already has rider information)
    
    # Rename columns
    
    predict_vector = predict_vector.rename(columns=feature_names)
    
    # Implement time conversion function
    
    predict_vector = time_conv(predict_vector)
    
    # Implement time diff function
    
    predict_vector = time_diffs(predict_vector)
    
    # Add Rider Experience based on Age Column - Low - Medium - High
    
    exp = []
    for i in list(predict_vector['Age']) :
        if i <= 495 :
            exp.append('low')
        elif i > 495 and i <= 1236 :
            exp.append('medium')
        else :
            exp.append('high')
            
    
    predict_vector['Rider_Exp'] = exp        
    
    # Filling Missing Values for Temperature and Precipitation - used the Mean
    
    
    predict_vector['Temperature'] = predict_vector['Temperature'].fillna(
                                                       23.255688596099763)
    
    
    predict_vector['Precipitation(mm)'] = predict_vector[
                                 'Precipitation(mm)'].fillna(
                                                          7.573501997336319)
    
    ## Create Temperature band Column - 3 categories - low, mid, high
    
    temp = []
    for i in list(predict_vector['Temperature']) :
        if i <= 21.4 :
            temp.append('low')
        elif i > 21.4 and i <= 25.3 :
            temp.append('medium')
        else :
            temp.append('high')
    
    predict_vector['Temp_Band'] = temp
    
    # Adding manhatten distance
    
    predict_vector= added_manhattan(predict_vector)
    
    # Implementing harvisine distance function
    
    predict_vector = add_haversine(predict_vector)
    
    # This is to check if there is any difference between
    # the columns with Days of Month or Weekday of Month
    
    month_cols = [col for col in predict_vector.columns if col.endswith('Mon')]
    weekday_cols = [col for col in predict_vector.columns if col.endswith('Weekday')]
    
    count = 0
    instances_of_different_days = [];
    for i, row in predict_vector.iterrows():
        if len(set(row[month_cols].values)) > 1:
            # print(count+1, end='\r')
            count = count + 1
            instances_of_different_days.append(list(row[month_cols].values))
    
    
    predict_vector['Day_of_Month'] = predict_vector[month_cols[0]]
    predict_vector['Day_of_Week'] = predict_vector[weekday_cols[0]]
    
    predict_vector.drop(month_cols+weekday_cols, axis=1, inplace=True)
    predict_vector.drop('Vehicle_Type', axis=1, inplace=True)
    
    #Convert Personal_Business Temp_Band using LabelEncoding
    
    le = LabelEncoder()
    le.fit(predict_vector['Personal_Business'])
    predict_vector['Personal_Business'] = le.transform(
                                    predict_vector['Personal_Business'])
    
    
    # Rider_Exp convert Label Encoding
    
    le.fit(predict_vector['Rider_Exp'])
    predict_vector['Rider_Exp'] = le.transform(predict_vector['Rider_Exp'])
    
    
    # Convert Temp_Band using LabelEncoding
    
    le.fit(predict_vector['Temp_Band'])
    predict_vector['Temp_Band'] = le.transform(predict_vector['Temp_Band'])
    
    numeric_cols = []
    object_cols = []
    time_cols = []
    for k, v in predict_vector.dtypes.items():
        if (v != object):
            if (k != "Time_Pic_Arr"):
                numeric_cols.append(k)
        elif k.endswith("Time"):
            time_cols.append(k)
        else:
            object_cols.append(k)
    
    
    features = numeric_cols 
    
    predict_vector = predict_vector[features]
    
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model : str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data , num_iteration=model.best_iteration)
    # Format as list for output standerdisation.
    return prediction[0].tolist()