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

def time_diffs(input_df):
    df = input_df.copy()
    df['Conf_Pla_dif'] = df['Con_Time'] - df['Pla_Time']
    df['Arr_Con_dif'] = df['Arr_Pic_Time'] - df['Con_Time']
    df['Pic_Arr_dif'] = df['Pickup_Time'] - df['Arr_Pic_Time']

    return df

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

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h
def add_haversine(input_df):
    input_df_1 = input_df.copy()
    input_df_1['distance_haversine'] = haversine_array(input_df_1['Pickup_Lat'].values,
                                                       input_df_1['Pickup_Lon'].values,
                                                       input_df_1['Destination_Lat'].values,
                                                       input_df_1['Destination_Lon'].values)
    return input_df_1


order_to_train =['Platform_Type',
                 'Pla_Mon',
                 'Pla_Weekday',
                 'Pla_Time',
                 'Con_Day_Mon',
                 'Con_Weekday',
                 'Con_Time',
                 'Arr_Pic_Mon',
                 'Arr_Pic_Weekday',
                 'Arr_Pic_Time',
                 'Pickup_Mon',
                 'Pickup_Weekday',
                 'Pickup_Time',
                 'Distance(km)',
                 'Temperature',
                 'Precipitation(mm)',
                 'Pickup_Lat',
                 'Pickup_Lon',
                 'Destination_Lat',
                 'Destination_Lon',
                 'No_Of_Orders',
                 'Age',
                 'Average_Rating',
                 'No_of_Ratings',
                 'Conf_Pla_dif',
                 'Arr_Con_dif',
                 'Pic_Arr_dif',
                 'manhattan_dist',
                 'distance_haversine',
                 'Rider_Exp_medium',
                 'Rider_Exp_high',
                 'Personal_Business_Personal',
                 'Temp_Band_medium',
                 'Temp_Band_high']




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

    
    predict_vector = feature_vector_df.copy()
    
    # Concatinate train and test(which already has rider information)
    
    # Rename columns
    
    predict_vector = predict_vector.rename(columns=feature_names)
    
    predict_vector = Impute(predict_vector)

    # Implement time diff function
    
    predict_vector = time_change(predict_vector)
    
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
                                                       23.3)
    
    
    predict_vector['Precipitation(mm)'] = predict_vector[
                                 'Precipitation(mm)'].fillna(
                                                          7.6)
    
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
    
    # Encoding Personal_Business Manually
    le = LabelEncoder()
    
    if str(predict_vector['Personal_Business']) == 'personal' :
           predict_vector['Personal_Business_Personal'] = [1]
    else :
           predict_vector['Personal_Business_Personal'] = [0]
           
           
    predict_vector = predict_vector.drop(['Personal_Business'], axis=1)
    
    
           
    # Rider_Exp Label Encoding
    
    if str(predict_vector['Rider_Exp']) == 'low' :
        predict_vector['Rider_Exp_medium'] = [0]
        predict_vector['Rider_Exp_high'] = [0]
        
    
    elif str(predict_vector['Rider_Exp']) == 'medium' :
        predict_vector['Rider_Exp_medium'] = [1]
        predict_vector['Rider_Exp_high'] = [0]
        
    else :
        predict_vector['Rider_Exp_medium'] = [0]
        predict_vector['Rider_Exp_high'] = [1]
        
    predict_vector = predict_vector.drop(['Rider_Exp'], axis=1)    
    
    # Temp_Band Label Encoding
    
    if str(predict_vector['Temp_Band']) == 'low' :
        predict_vector['Temp_Band_medium'] = [0]
        predict_vector['Temp_Band_high'] = [0]
        
    
    elif str(predict_vector['Temp_Band']) == 'medium' :
        predict_vector['Temp_Band_medium'] = [1]
        predict_vector['Temp_Band_high'] = [0]
        
    else :
        predict_vector['Temp_Band_medium'] = [0]
        predict_vector['Temp_Band_high'] = [1]
        
    predict_vector = predict_vector.drop(['Temp_Band'], axis=1)
    
    # Various distance function implementation
    
    predict_vector = time_diffs(predict_vector)
    
    predict_vector = manhattan(predict_vector)
    
    predict_vector = add_haversine(predict_vector)
    
    # Encode Rider Exp,Temp_Band and Personal/Business
    
    # Extract feature columns
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
            
            
    predict_vector = predict_vector[numeric_cols]
    predict_vector = predict_vector[order_to_train]
    
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