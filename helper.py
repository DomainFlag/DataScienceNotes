import pandas as pd
import numpy as np

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from pandas.core.dtypes.common import is_datetime_or_timedelta_dtype


def split_date(data_frame, col_name, alias):
    ''' helper function to split the date into corresponding categoricals '''
    
    # split date into corresponding features
    split_date.props = ["year", "month", "day", "dayofweek"]
    
    # extract date column
    col_date = data_frame[col_name]
    
    assert(is_datetime_or_timedelta_dtype(col_date.values))
    
    # extracting date properties and storing into individual columns
    for feature in split_date.props:
        
        # check if it contains a datetime property
        if(hasattr(col_date.dt, feature)):
            
            # create a feature as [alias]_[feature_name]
            data_frame[f"{alias}_{feature}"] = getattr(col_date.dt, feature)
            
    # extracting timestamp
    data_frame[f"{alias}_timestamp"] = col_date.astype(np.int64) // (10 ** 9)
    
    # removing the raw column
    data_frame.drop(columns = [col_name], inplace = True)
    
    
def trans_categorical(data_frame, labels = []):
    ''' helper function to tranform text-based columns to numerical(categorical ones)'''
    
    # loop through each col
    for label, column in data_frame.items():
        
        if(label in labels):
            
            # columns/labels to avoid
            continue
        
        if(is_string_dtype(data_frame[label])):
            
            # if string convert as category
            data_frame[label] = data_frame[label].astype("category")
            
            
def trans_numerical(data_frame, target, suffle_data_frame = True):
    ''' helper function for retrieving numerical features and targets, 
        null values are transformed into the median value of the column. 
    '''

    # copy the data frame
    data_frame_c = data_frame.copy()
    
    if(suffle_data_frame):
        
        # shuffle rows as we are dealing with data non-sequence
        data_frame_c = data_frame_c.sample(frac = 1).reset_index(drop = True)

    # loop through each col and normalize
    for label, column in data_frame_c.items():
        
        # check if numerical and contain nulls
        if(is_numeric_dtype(column) and pd.isnull(column).sum() != 0):
            
            # column median
            median = column.median()
            
            # fill the data with the median
            data_frame_c[label] = column.fillna(median)
            
    # loop through each col and change to numerical
    for label, column in data_frame_c.items(): 
        
        # if numerical nothing to do
        if(not is_numeric_dtype(column) and is_categorical_dtype(column)):
            
            # change to numerical data
            data_frame_c[label] = data_frame_c[label].astype("category").cat.codes + 1
    
    return [ data_frame_c.drop(columns = [ target ]), pd.Series(data_frame_c[target].values) ]


def split_dataset(data, threshold, columns = None):
    ''' helper function to split data into training and validation set '''
    
    # provisional data copied
    data_prov = data.copy()
    
    if(columns is not None):
        
        # filter columns
        data_prov = data_prov[columns]

    return data_prov[threshold:], data_prov[:threshold]

def split_data(features, targets, threshold, columns = None, subset = None):
    ''' helper function to split data into training and validation set '''
    
    assert(threshold >= 0 and threshold <= 1.0)
    
    if(subset is None):
        
        # subset is our full set
        subset = targets.size
        
    else:
        
        # smaller set of data
        features, targets = features[:subset], targets[:subset]
        
    # update our threeshold for current subset
    threshold = int(threshold * subset)
    
    # splitting features
    train_features, valid_features = split_dataset(features, threshold, columns = columns)
    
    # splitting targets
    train_targets, valid_targets = split_dataset(targets, threshold, columns = None)
    
    return (train_features, train_targets), (valid_features, valid_targets)