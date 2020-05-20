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

# Re-ordering columns ( Hard code )

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
    
    # Drop useless columns
    
    predict_vector = feature_vector_df.drop([
                    'Vehicle Type','Precipitation in millimeters' ,
                    'User Id','Order No','Rider Id' ,'Platform Type',
                    'Personal or Business',
                     ],axis = 1)
    
    # Populate temperature nulls
    
    # predict_vector = predict_vector.fillna(predict_vector.mean())
      
    
    # Time manipulation application
    
    predict_vector['Placement - Hour'] = predict_vector[
                                      'Placement - Time'
                                      ].apply(time_parse_hr)
    
    predict_vector['Placement - Minute'] = predict_vector[
                                     'Placement - Time'
                                     ].apply(time_parse_min)
    
    predict_vector['Placement - Second'] = predict_vector[
                                     'Placement - Time'
                                     ].apply(time_parse_sec)
                                                                
    predict_vector['Confirmation - Hour'] = predict_vector[
                                   'Confirmation - Time'
                                   ].apply(time_parse_hr)
    
    predict_vector['Confirmation - Minute'] = predict_vector[
                                   'Confirmation - Time'
                                   ].apply(time_parse_min)
    
    predict_vector['Confirmation - Second'] = predict_vector[
                                         'Confirmation - Time'
                                     ].apply(time_parse_sec)
    
    predict_vector['Arrival at Pickup - Hour'] = predict_vector[
                                        'Arrival at Pickup - Time'
                                            ].apply(time_parse_hr)
    
    predict_vector['Arrival at Pickup - Minute'] = predict_vector[
                                        'Arrival at Pickup - Time'
                                              ].apply(time_parse_min)
    
    predict_vector['Arrival at Pickup - Second'] = predict_vector[
                                        'Arrival at Pickup - Time'
                                              ].apply(time_parse_sec)
    
    predict_vector['Pickup - Hour'] = predict_vector[
                                     'Pickup - Time'
                                      ].apply(time_parse_hr)
    
    predict_vector['Pickup - Minute'] = predict_vector[
                                     'Pickup - Time'
                                     ].apply(time_parse_min)
    
    predict_vector['Pickup - Second'] = predict_vector[
                                       'Pickup - Time'
                                       ].apply(time_parse_sec)


    # Drop original times
    
    predict_vector = predict_vector.drop(['Placement - Time' ,
                                          'Confirmation - Time',
                                          'Arrival at Pickup - Time' ,
                                          'Pickup - Time'],
                                          axis = 1)
    
    predict_vector = predict_vector[columns_to_keep] # Define the order
       
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
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()