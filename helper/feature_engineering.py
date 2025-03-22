import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

from math import ceil

def create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    copy_df = df.copy()
    
    # Ensure the DataFrame index is a datetime index
    if not isinstance(copy_df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    
    # Extract various datetime features
    # tm_y: This feature represents the year (normalized).
    # tm_wm: This feature represents the week of the month.
    # tm_dw: This feature represents the day of the week, where Monday is 0 and Sunday is 6.
    copy_df['tm_d'] = copy_df.index.day.astype(np.int8)
    copy_df['tm_w'] = copy_df.index.isocalendar().week.astype(np.int8)
    copy_df['tm_m'] = copy_df.index.month.astype(np.int8)
    copy_df['tm_y'] = (copy_df.index.year - copy_df.index.year.min()).astype(np.int8)
    copy_df['tm_wm'] = copy_df['tm_d'].apply(lambda x: ceil(x / 7)).astype(np.int8)
    copy_df['tm_dw'] = copy_df.index.dayofweek.astype(np.int8)
    copy_df['tm_w_end'] = (copy_df['tm_dw'] >= 5).astype(np.int8)
    
    # Create features for the hour of the day and the half-hour of the day
    copy_df['hour_of_day'] = copy_df.index.hour.astype(np.int8)
    copy_df['halfhour_of_day'] = (copy_df.index.hour * 2 + copy_df.index.minute // 30).astype(np.int8)
    
    # Create features for the EFA block of the day
    efa_block_start = 23  # EFA blocks start at 23:00
    efa_block_duration = 4  # EFA blocks last 4 hours
    efa_blocks_per_day = 24 // efa_block_duration
    
    copy_df['efa_block'] = ((((copy_df.index.hour - efa_block_start) % 24) // efa_block_duration)+1).astype(np.int8)
    
    return copy_df[['tm_d', 'tm_w', 'tm_m', 'tm_y', 'tm_wm', 'tm_dw', 'tm_w_end', 'hour_of_day', 'halfhour_of_day', 'efa_block']]

def create_lagged_features(df: pd.DataFrame, target:str, lag_days:list, drop_target:bool) -> pd.DataFrame:
    # Create a temporary DataFrame with the "id" column and the target column
    temp_df = df[[target]]
    
    # Create lagged features for the target column
    temp_df = temp_df.assign(**{
        '{}_lag_{}'.format(col, l): temp_df[col].transform(lambda x: x.shift(l))
        for l in lag_days
        for col in [target]
    })
    # Drop the target column if the drop_target parameter is set to True
    if drop_target:
        return temp_df.drop(columns=[target])
    else:
        return temp_df

def create_sincos_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    # Will take features for the time of week, day, hour of day, halfhour of day and EFA block and turn them into Sinusoidal features, it will also drop the original features
    """
    Needs to be checked if this was done correctly, this is a good source to understand what's happening: https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/
    """
    
    copy_df = df.copy()

    # Ensure the DataFrame index is a datetime index
    if not isinstance(copy_df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    
    # Sinusoidal features for the time of week, day, hour of day, halfhour of day and EFA block
    max_tm_wm = copy_df['tm_wm'].max()
    copy_df['sin_tm_wm'] = np.sin(2 * np.pi * copy_df['tm_wm'] / max_tm_wm)
    copy_df['cos_tm_wm'] = np.cos(2 * np.pi * copy_df['tm_wm'] / max_tm_wm)

    copy_df['sin_tm_dw'] = np.sin(2 * np.pi * copy_df['tm_dw'] / 6)  # 6 is highest value of tm_dw (0-6)
    copy_df['cos_tm_dw'] = np.cos(2 * np.pi * copy_df['tm_dw'] / 6)

    copy_df['sin_hour_of_day'] = np.sin(2 * np.pi * copy_df['hour_of_day'] / 23)
    copy_df['cos_hour_of_day'] = np.cos(2 * np.pi * copy_df['hour_of_day'] / 23)

    copy_df['sin_halfhour_of_day'] = np.sin(2 * np.pi * copy_df['halfhour_of_day'] / 47)
    copy_df['cos_halfhour_of_day'] = np.cos(2 * np.pi * copy_df['halfhour_of_day'] / 47)

    max_efa_block = copy_df['efa_block'].max()
    copy_df['sin_efa_block'] = np.sin(2 * np.pi * copy_df['efa_block'] / max_efa_block)
    copy_df['cos_efa_block'] = np.cos(2 * np.pi * copy_df['efa_block'] / max_efa_block)

    return copy_df.drop(columns=['tm_wm', 'tm_dw', 'hour_of_day', 'halfhour_of_day', 'efa_block'])

def create_all_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    # Define the target columns and their respective lag days
    targets_and_lags = {
        'Day Ahead Price (EPEX half-hourly, local) - GB (£/MWh)': [48, 96],
        'Ancillary Price - DC-H - GB (£/MW/h)': [48, 96, 144, 192, 240, 288, 336],
        'Ancillary Price - DC-L - GB (£/MW/h)': [48, 96, 144, 192, 240, 288, 336],
        'Ancillary Price - DM-H - GB (£/MW/h)': [48, 96, 144, 192, 240, 288, 336],
        'Ancillary Price - DM-L - GB (£/MW/h)': [48, 96, 144, 192, 240, 288, 336],
        'Ancillary Price - DR-H - GB (£/MW/h)': [48, 96, 144, 192, 240, 288, 336],
        'Ancillary Price - DR-L - GB (£/MW/h)': [48, 96, 144, 192, 240, 288, 336],
        'Ancillary Volume Accepted - DC-H - GB (MW)': [48, 96, 144, 192, 240, 288, 336],
        'Ancillary Volume Accepted - DC-L - GB (MW)': [48, 96, 144, 192, 240, 288, 336],
        'Ancillary Volume Accepted - DM-H - GB (MW)': [48, 96, 144, 192, 240, 288, 336],
        'Ancillary Volume Accepted - DM-L - GB (MW)': [48, 96, 144, 192, 240, 288, 336],
        'Ancillary Volume Accepted - DR-H - GB (MW)': [48, 96, 144, 192, 240, 288, 336],
        'Ancillary Volume Accepted - DR-L - GB (MW)': [48, 96, 144, 192, 240, 288, 336],
        'Day Ahead Price (N2EX, local) - GB (£/MWh)': [48],
        'Day Ahead Price (EPEX, local) - GB (£/MWh)': [48]
    }
    
    # Create lagged features for each target column
    lagged_dfs = []
    for target, lag_days in targets_and_lags.items():
        # drop_target will be True if target column not in [day ahead hourly N2EX, day ahead hourly EPEX] 
        drop_target = target not in ['Day Ahead Price (N2EX, local) - GB (£/MWh)', 'Day Ahead Price (EPEX, local) - GB (£/MWh)']
        lagged_df = create_lagged_features(df, target, lag_days, drop_target)
        lagged_dfs.append(lagged_df)
    
    # Concatenate all the lagged DataFrames
    df_merged_lag = pd.concat(lagged_dfs, axis=1)
    
    return df_merged_lag
