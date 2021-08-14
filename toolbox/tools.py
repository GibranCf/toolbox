import pandas as pd 
import numpy as np
import re
import os 
from datetime import datetime
from pathlib import Path as path
import warnings
import time
import functools
from itertools import chain, islice, cycle

# function to find files inside a path by a pattern
def list_files(path2Data,pattern='.'):
    """
    path2Data  --  Pathlib path class object. 
                   path2Data = pathlib.path('users/Documents/folder')
    pattern    --  Regex pattern to match de file. 
                   Example, to find all csvs pattern = '\.csv$'
    """
    files = []
    for root,folder,filelist in os.walk(path2Data):
        for file in filelist:
            matchfile = '/'.join([root,file]) if re.compile(pattern).search(file) is not None else None
            if matchfile is not None:
                files.append(matchfile)
    if len(files) > 1: 
        warnings.warn('More than one file matched the pattern ')
def to_date(string, format):
    """
    Passes string to datetime object given date format 
    example = '2021-02-01' with format '%Y-%m-%d'
    """
    try:
        dttm = datetime.strptime(string,format)
        return dttm
    except:
        return pd.NaT
def to_float(x):
    """
    Pass string to float. Useful when numbers have commas, currency symbols etc.
    """
    x = str(x)
    x = re.compile('\d+\.{0,1}\d*').search(x)
    return float(x.group(0) if x is not None else np.nan)
def to_int(x):
    """
    Pass string to integers. Useful when numbers have commas, currency symbols etc.
    If float it's equivalent to floor at integer level
    """
    x = str(x)
    x = re.compile('\d+').search(x)
    return int(x.group(0) if x is not None else np.nan)
def twistDt(dt,pivot_cols,variable_name,column_name,
            col_format=None,var_format=None,varkwargs={},colkwargs={}):
    """
    Function to take dataframes from horizontal human readable format to vertical format.
    Helpful when handling time series data un horizontal format

    dt     --  Dataframe
    pivot_cols  -- list of column(s) that will remained fixed, the rest will be transposed
    column_name -- variable name of the new twisted data frame where the current colnames will 
                   be stored. In time series usually the dates
    var_name    -- variable name of the new twisted data frame where the current column values 
                   will be stored. In time series, the values across time
    col_format  -- callable function to format the current column names, when dates useful to apply a datetime
                   from string transformation
    var_format  -- callable function to format the column values
    colkwargs   -- kwargs for the format function
    varkwargs   -- kwargs for the format function
    """
    cols = list(dt.columns)
    for c in pivot_cols:
        cols.remove(c)
    df = pd.DataFrame()
    for col in cols:
        tdt = dt[pivot_cols + [col]]
        tdt.rename(columns = {col:variable_name},inplace=True)
        tdt[column_name] = col
        df = df.append(tdt,ignore_index=True)
    if col_format is not None:
        df[column_name] = df[column_name].apply(lambda x: col_format(x,**colkwargs))
    if var_format is not None:
        df[variable_name] = df[variable_name].apply(lambda x: var_format(x,**varkwargs))
    return df
def expandDt(data,axis,expvals,expcol):
    """
    Function to expand data frame. Over the axis it takes the max lower available variable data point in expcol column and expands it
    across the expvals that are below this point. if expavals > max data point, the last value will be taken to expand.
    Useful to expand date configuration files 

    dt       --  Data frame to expand
    axis     --  Axis or subset of cols of dt that will group the data to be expanded
    expvals  --  Values across wich the axis groups will be expanded
    expcol   --  column that will be contrasted against the expvals 
    """
    expdt = pd.DataFrame()
    gdata = data.groupby(axis)
    for index, df in gdata:
        df.sort_values([expcol]+axis,inplace=True)
        df['value_' +  expcol] = df[expcol]
        valsdt = pd.DataFrame()
        valsdt[expcol] = expvals
        df = pd.merge(valsdt,df,how='outer',indicator=True).sort_values(expcol)
        datevals = dt[[expcol,'value_'+expcol]].drop_duplicates().fillna(method='ffill').fillna(method='backfill')
        dt = pd.merge(datevals,df.drop(columns=expcol)).drop(columns='value_'+expcol)
        expdt = expdt.append(dt,ignore_index=False)
    return expdt
def expandDiscDt(data,expand_dt,axis_cols,expand_axis):
    axisdt = data[axis_cols].drop_duplicates()
    axisdt['dummy'] = '1'
    data_axis = pd.DataFrame()
    for i,y in expand_dt.groupby(expand_axis):
        x = pd.merge(axisdt[expand_axis].drop_duplicates(),y)
        df = y.copy()
        df['dummy'] = '1'
        x['dummy'] = '1'
        df = pd.merge(x,df,how='outer').drop(columns='dummy')
        data_axis= data_axis.append(df,ignore_index=True)
    fulldata=pd.merge(data_axis,data,how='outer',indicator=True).sort_values(axis_cols)
    return fulldata

