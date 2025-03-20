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
    
    return copy_df

def create_lagged_features(df: pd.DataFrame, target:str, lag_days:list, drop_target:bool) -> pd.DataFrame:
    # Create a temporary DataFrame with the "id" column and the target column
    temp_df = df[[target]]
    
    # Create lagged features for the target column
    temp_df = temp_df.assign(**{
        '{}_lag_{}'.format(col, l): temp_df[col].transform(lambda x: x.shift(l))
        for l in lag_days
        for col in [target]
    })
    # Merge temporary DataFrame with the original DataFrame
    if drop_target:
        return temp_df.drop(columns=[target]).merge(df, how='left', left_index=True, right_index=True)
    else:
        return temp_df.merge(df, how='left', left_index=True, right_index=True)